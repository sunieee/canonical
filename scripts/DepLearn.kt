package tarmorn

import tarmorn.data.IdManager
import tarmorn.data.RelationPath
import tarmorn.data.TripleSet
import tarmorn.eval.HitsAtK
import tarmorn.eval.ResultSet
import tarmorn.structure.TLearn.DepAtom
import tarmorn.structure.TLearn.Metric
import tarmorn.structure.TLearn.RuleParser
import tarmorn.data.MyTriple
import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.InputStreamReader
import java.io.PrintWriter
import java.nio.charset.StandardCharsets
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * DepLearn - Dependency-based learning algorithm
 * 
 * This class reads a TripleSet and rule file, builds indexes and converts rules to H2B2metric using DepAtom.
 * 
 * Key features:
 * 1. Builds indexes from TripleSet:
 *    - r2h2tSet: relation -> head -> tails mapping
 *    - r2instanceSet: relation -> set of (head, tail) pairs as Long
 *    - r2tSet: relation -> array of tail entities
 * 
 * 2. Reads rules from PATH_RULES (excludes complex rules with &&)
 *    Rule format: bodySize\tsupport\tconfidence\thead <= body1 body2 ...
 * 
 * 3. Converts rules to H2B2metric structure:
 *    - Uses DepAtom instead of MyAtom (no instance storage, direct index lookup)
 *    - H2B2metric: ConcurrentHashMap<DepAtom, ConcurrentHashMap<DepAtom, Metric>>
 *    - headSize is obtained directly from indexes (no need to store in DepAtom)
 * 
 * Usage:
 *   mvn compile
 *   mvn exec:java -Dexec.mainClass="tarmorn.DepLearn"
 * 
 * @see DepAtom for atom representation without instance storage
 * @see TLearn for comparison with the full learning algorithm
 */
object DepLearn {
    
    // Core data structures from TripleSet
    lateinit var ts: TripleSet
    lateinit var r2h2tSet: Map<Long, Map<Int, Set<Int>>>
    lateinit var r2instanceSet: Map<Long, Set<Long>>
    lateinit var r2tSet: Map<Long, IntArray>
    
    // Rule metric structure: head -> body -> ruleId
    val H2B2ID = ConcurrentHashMap<DepAtom, ConcurrentHashMap<DepAtom, Int>>()
    val ID2metric = ConcurrentHashMap<Int, Metric>()
    val depAdj2metric = ConcurrentHashMap<Int, ConcurrentHashMap<Int, Metric>>()
    val ruleId2HeadRelationId = ConcurrentHashMap<Int, Long>()
    
    // Statistics variables
    var totalRules = 0
    // Lift statistics for composition phase
    val unaryPositiveLift = java.util.concurrent.atomic.AtomicInteger(0)
    val unaryNegativeLift = java.util.concurrent.atomic.AtomicInteger(0)
    val binaryPositiveLift = java.util.concurrent.atomic.AtomicInteger(0)
    val binaryNegativeLift = java.util.concurrent.atomic.AtomicInteger(0)
    val mixPositiveLift = java.util.concurrent.atomic.AtomicInteger(0)
    val mixNegativeLift = java.util.concurrent.atomic.AtomicInteger(0)
    val thread0Attempts = java.util.concurrent.atomic.AtomicInteger(0)
    
    // Constants from TLearn
    const val MIN_SURPRISAL_LIFT = 0.05
    const val TOP_K_RULE_COMBO = 300

    private fun format5(value: Double): String {
        val formatted = String.format(java.util.Locale.US, "%.5f", value)
        return formatted.trimEnd('0').trimEnd('.')
    }

    private fun packBinaryInstance(head: Int, tail: Int): Long {
        return (head.toLong() shl 32) or (tail.toLong() and 0xFFFFFFFFL)
    }

    private fun storeDependency(
        ruleId1: Int,
        ruleId2: Int,
        metric: Metric,
        metric1: Metric,
        metric2: Metric,
        positiveCounter: java.util.concurrent.atomic.AtomicInteger,
        negativeCounter: java.util.concurrent.atomic.AtomicInteger
    ) {
        var id1 = ruleId1
        var id2 = ruleId2
        var m1 = metric1
        var m2 = metric2
        var conf1 = m1.confidence
        var conf2 = m2.confidence
        if (conf1 < conf2) {
            val tmpId = id1
            id1 = id2
            id2 = tmpId
            val tmpMetric = m1
            m1 = m2
            m2 = tmpMetric
            val tmpConf = conf1
            conf1 = conf2
            conf2 = tmpConf
        }

        val lift = metric.surprisal - m1.surprisal - m2.surprisal
        val maxSurprisal = maxOf(m1.surprisal, m2.surprisal)
        if (lift > 0 || metric.surprisal < maxSurprisal) {
            metric.lift = if (lift > 0) lift else metric.surprisal - maxSurprisal
            val inner = depAdj2metric.computeIfAbsent(id1) { ConcurrentHashMap() }
            inner[id2] = metric
            if (metric.lift > 0) positiveCounter.incrementAndGet()
            else negativeCounter.incrementAndGet()
        }
    }

    private fun escapeJson(value: String): String {
        return value.replace("\\", "\\\\").replace("\"", "\\\"")
    }

    private fun buildBodyList(bodyMap: ConcurrentHashMap<DepAtom, Int>): List<Triple<DepAtom, Int, Metric>> {
        return bodyMap.entries
            .mapNotNull { entry ->
                val metric = ID2metric[entry.value] ?: return@mapNotNull null
                if (metric.surprisal >= MIN_SURPRISAL_LIFT && metric.surprisal < Settings.MAX_SURPRISAL) {
                    Triple(entry.key, entry.value, metric)
                } else null
            }
            .sortedByDescending { it.third.confidence }
            .take(TOP_K_RULE_COMBO)
    }

    private fun precomputeBodyLists(
        threadPool: java.util.concurrent.ExecutorService
    ): ConcurrentHashMap<DepAtom, List<Triple<DepAtom, Int, Metric>>> {
        val bodyListMap = ConcurrentHashMap<DepAtom, List<Triple<DepAtom, Int, Metric>>>()
        val futures = H2B2ID.entries.map { (headAtom, bodyMap) ->
            threadPool.submit {
                val bodyList = buildBodyList(bodyMap)
                bodyListMap[headAtom] = bodyList

                bodyList.forEach { (bodyAtom, _, _) ->
                    if (bodyAtom.isBinary && !bodyAtom.isL1Atom && !bodyAtom.hasBeenSampled) {
                        synchronized(bodyAtom) {
                            if (!bodyAtom.hasBeenSampled && !bodyAtom.isL1Atom) {
                                bodyAtom.sampleBinaryInstancesEDIS()
                            }
                        }
                    }
                }
            }
        }
        futures.forEach { it.get() }
        return bodyListMap
    }

    
    /**
     * Main entry point
     */
    @JvmStatic
    fun main(args: Array<String>) {
        Settings.load()
        println("DepLearn - Dependency-based learning algorithm")
        println("=".repeat(60))
        
        val startTime = System.currentTimeMillis()
        
        // Step 1: Load triple set and build indexes
        println("\n=== Step 1: Loading Triple Set ===")
        println("Loading triple set from: ${Settings.PATH_TRAINING}")
        loadTripleSet()

        readRules(Settings.PATH_RULES)

        saveMetricToJson(
            metricMap = H2B2ID,
            outputPath = Settings.PATH_H2B2metric,
            appendMode = false,
            isFormulaMap = false
        )

        try {
            compositionPhase()
        } catch (e: Exception) {
            println("Error during composition phase: ${e.message}")
            e.printStackTrace()
        }
        printStatistics()

        saveDependencyToFile(Settings.PATH_DEPENDENCY)

        val endTime = System.currentTimeMillis()
        val elapsedSeconds = (endTime - startTime) / 1000.0
        
        println("\n" + "=".repeat(60))
        println("DepLearn completed in ${"%.2f".format(elapsedSeconds)}s")
    }
    
    /**
     * Load triple set and build indexes
     */
    private fun loadTripleSet() {
        ts = TripleSet(Settings.PATH_TRAINING, true)
        
        // Build r2h2tSet: relation -> head -> tails
        r2h2tSet = ts.r2h2tSet
        
        // Build r2instanceSet: relation -> set of (h,t) pairs as Long
        r2instanceSet = r2h2tSet.mapValues { (_, h2tSet) ->
            h2tSet.flatMap { (head, tails) ->
                tails.map { tail -> 
                    (head.toLong() shl 32) or (tail.toLong() and 0xFFFFFFFFL) 
                }
            }.toSet()
        }
        
        // Build r2tSet: relation -> array of tail entities (for inverse relation)
        r2tSet = r2h2tSet.keys.associateWith { r ->
            val inv = RelationPath.getInverseRelation(r)
            val keys = r2h2tSet[inv]?.keys ?: emptySet()
            keys.toIntArray()
        }
        
        println("Loaded triple set: ${ts.size} triples")
        println("Relations: ${r2h2tSet.keys.size}")
        println("Indexed r2h2tSet, r2instanceSet, r2tSet")
    }
    
    /**
     * Read rules from file and convert to H2B2metric
     * Rule format: bodySize\tsupport\tconfidence\thead <= body1 body2 ...
     * We only process rules without && (no complex rules)
     */
    private fun readRules(filepath: String) {
        println("\n=== Step 2: Reading Rules ===")
        println("Reading rules from: $filepath")
        val file = File(filepath)
        if (!file.exists()) {
            println("Warning: Rule file not found: $filepath")
            return
        }
        
        val startTime = System.currentTimeMillis()
        
        // First pass: read all lines into memory
        val allLines = mutableListOf<Pair<Int, String>>()
        BufferedReader(InputStreamReader(FileInputStream(file), StandardCharsets.UTF_8)).use { reader ->
            var lineNumber = 0
            reader.forEachLine { line ->
                lineNumber++
                if (line.isNotBlank() && !line.startsWith("#")) {
                    val tokens = line.split("\t")
                    if (tokens.size >= 4) {
                        allLines.add(lineNumber to line)
                    }
                }
            }
        }
        
        println("Read ${allLines.size} valid rules, starting parallel parsing...")
        
        // Concurrent counters
        val parsedRules = java.util.concurrent.atomic.AtomicInteger(0)
        val errors = java.util.concurrent.atomic.AtomicInteger(0)
        
        // Create thread pool
        val threadPool = java.util.concurrent.Executors.newFixedThreadPool(Settings.WORKER_THREADS)
        
        try {
            // Chunk the lines for better load balancing
            val chunkSize = maxOf(1000, allLines.size / (Settings.WORKER_THREADS * 4))
            val chunks = allLines.chunked(chunkSize)
            
            val futures = chunks.map { chunk ->
                threadPool.submit {
                    chunk.forEach { (lineNumber, line) ->
                        try {
                            parseAndAddRule(line, lineNumber)
                            val count = parsedRules.incrementAndGet()
                            if (count % 100000 == 0) {
                                println("Parsed $count/${allLines.size} rules...")
                            }
                        } catch (e: Exception) {
                            val errorCount = errors.incrementAndGet()
                            if (errorCount <= 10) {
                                synchronized(System.out) {
                                    println("Error parsing rule: ${e.message}")
                                    println("  Line: $line")
                                }
                            }
                        }
                    }
                }
            }
            
            // Wait for all tasks to complete
            futures.forEach { it.get() }
            
        } finally {
            threadPool.shutdown()
        }
        
        val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
        println("\nRule loading completed:")
        println("  Total lines processed: ${allLines.size}")
        println("  Successfully parsed: ${parsedRules.get()}")
        println("  Errors: ${errors.get()}")
        println("  Time: %.2f seconds".format(elapsed))
        println("  Speed: %.0f rules/sec".format(parsedRules.get() / elapsed))
    }

    fun setH2B2ID(headAtom: DepAtom, bodyAtom: DepAtom, ruleId: Int, metric: Metric) {
        val B2id = H2B2ID.computeIfAbsent(headAtom) { ConcurrentHashMap() }
        B2id[bodyAtom] = ruleId
        ID2metric[ruleId] = metric
        totalRules++
    }
    
    /**
     * Parse a single rule line and add to H2B2metric
     * Format: bodySize\tsupport\tconfidence\thead <= body
     * 
     * Note: body is a relation path string like "r1*r2(const)" or "r1*INVERSE_r2"
     * It will be parsed as a single DepAtom with encoded relationId
     */
    private fun parseAndAddRule(line: String, ruleId: Int) {
        val tokens = line.split("\t")
        if (tokens.size < 4) {
            throw IllegalArgumentException("Invalid rule format: expected at least 4 tokens")
        }
        
        val bodySize = tokens[0].toInt()
        val support = tokens[1].toDouble()
        val confidence = tokens[2].toDouble()
        val ruleString = tokens[3]

        if (ruleString.contains("&&")) {
            return
        }

        val (headAtom, bodyAtom) = RuleParser.parseRule(ruleString)
        

        val headSize = getAtomSize(headAtom)
        val metric = Metric(support, headSize, bodySize)

        if (bodyAtom != null) {
            setH2B2ID(headAtom, bodyAtom, ruleId, metric)
            ruleId2HeadRelationId[ruleId] = headAtom.relationId
        }
    }
    
    /**
     * Get the size (support) of an atom from indexes
     */
    private fun getAtomSize(atom: DepAtom): Int {
        return when {
            // Binary atom: relationId with Y
            atom.entityId == IdManager.getYId() -> {
                r2instanceSet[atom.relationId]?.size ?: 0
            }
            // Loop atom: relationId with X
            atom.entityId == IdManager.getXId() -> {
                ts.r2loopSet[atom.relationId]?.size ?: 0
            }
            // Existence atom: relationId with *
            atom.entityId == 0 -> {
                r2h2tSet[atom.relationId]?.keys?.size ?: 0
            }
            // Constant atom: relationId with specific entity
            else -> {
                r2h2tSet[atom.relationId]?.get(atom.entityId)?.size ?: 0
            }
        }
    }
    
    /**
     * Composition Phase - combine frequent atom sets using Eclat algorithm
     * Called after reading rules, builds formulas based on H2B2metric
     */
    fun compositionPhase() {
        println("\n=== Composition Phase ===")
        println("Starting Composition Phase...")
        
        val processedTasks = java.util.concurrent.atomic.AtomicInteger(0)
        val threadPool = java.util.concurrent.Executors.newFixedThreadPool(Settings.WORKER_THREADS)
        val compositionActiveThreadCount = java.util.concurrent.atomic.AtomicInteger(0)
        val compositionThreadMonitorLock = Object()
        
        try {
            val bodyListMap = precomputeBodyLists(threadPool)
            val workQueue = ConcurrentLinkedQueue<Pair<DepAtom, Int>>()
            var totalTasks = 0
            bodyListMap.forEach { (headAtom, bodyList) ->
                for (i in bodyList.indices) {
                    workQueue.add(headAtom to i)
                    totalTasks++
                }
            }

            val futures = (0 until Settings.WORKER_THREADS).map { _ ->
                threadPool.submit {
                    compositionActiveThreadCount.incrementAndGet()
                    try {
                        while (true) {
                            val task = workQueue.poll() ?: break
                            val headAtom = task.first
                            val i = task.second
                            val bodyList = bodyListMap[headAtom].orEmpty()
                            if (bodyList.isEmpty()) continue

                            if (headAtom.isBinary) {
                                processBinaryHeadAtom(headAtom, bodyList, i)
                            } else {
                                val binaryHeadAtom = headAtom.getBinaryAtom()
                                val binaryBodyList = bodyListMap[binaryHeadAtom].orEmpty()
                                processUnaryHeadAtom(headAtom, bodyList, binaryBodyList, i)
                            }

                            val cnt = processedTasks.incrementAndGet()
                            if (cnt % 10000 == 0) {
                                println("Processed $cnt/$totalTasks tasks...")
                            }
                        }
                    } finally {
                        val activeCount = compositionActiveThreadCount.decrementAndGet()
                        synchronized(compositionThreadMonitorLock) {
                            compositionThreadMonitorLock.notifyAll()
                        }
                    }
                }
            }

            // Monitor thread activity
            var lastActiveCount = 0
            while (true) {
                val activeCount: Int
                synchronized(compositionThreadMonitorLock) {
                    // Wait for thread count changes
                    while (compositionActiveThreadCount.get() == lastActiveCount && !futures.all { it.isDone }) {
                        compositionThreadMonitorLock.wait(1000)
                    }
                    activeCount = compositionActiveThreadCount.get()
                    lastActiveCount = activeCount
                }
                
                if (futures.all { it.isDone }) {
                    println("All composition tasks completed")
                    break
                }
                
                if (activeCount > 0 && activeCount < Settings.WORKER_THREADS - 5) {
                    println("Composition thread count: $activeCount/${Settings.WORKER_THREADS} active")
                }
                
                if (activeCount < 4 && activeCount > 0) {
                    println("FORCING SHUTDOWN: Less than 4 threads remaining in composition phase")
                    futures.forEach { it.cancel(true) }
                    threadPool.shutdownNow()
                    break
                }
            }
            
        } catch (e: Exception) {
            println("Error in composition phase monitoring: ${e.message}")
            threadPool.shutdownNow()
        } finally {
            threadPool.shutdown()
            threadPool.awaitTermination(1, java.util.concurrent.TimeUnit.HOURS)
        }
        
        val totalDeps = depAdj2metric.values.sumOf { it.size }
        println("Composition Phase completed. Total dependencies: $totalDeps")
    }
    
    /**
     * Process single binary headAtom, perform pairwise combination of bodyAtoms
     * Uses dynamic sampling strategy to handle large instance sets
     */
    private fun processBinaryHeadAtom(
        headAtom: DepAtom,
        bodyList: List<Triple<DepAtom, Int, Metric>>,
        i: Int
    ) {
        if (bodyList.size < 2) return  // Need at least 2 bodyAtoms to combine
        if (i !in bodyList.indices) return
        
        // 获取当前线程ID，用于控制日志输出（仅线程0输出详细日志）
        val threadId = Thread.currentThread().id % Settings.WORKER_THREADS
        val shouldDebug = (threadId == 0L) && thread0Attempts.incrementAndGet() <= 10
        
        var pairCount = 0
        val headInstances = headAtom.getBinaryInstances()

        // Pairwise combination with dynamic sampling (fixed i)
        val (B1, ruleId1, metric1) = bodyList[i]
        var S_H1_size = B1.instances.count { it in headInstances }
        val initialB1Size = B1.instances.size

        // Sample B1 until S_H1.size >= MIN_SUPP or exhausted
        // 只对非L1原子进行采样
        if (S_H1_size < Settings.MIN_SUPP && !B1.isL1Atom && !B1.samplingExhausted) {
            synchronized(B1) {
                S_H1_size = B1.instances.count { it in headInstances }
                while (S_H1_size < Settings.MIN_SUPP && !B1.isL1Atom && !B1.samplingExhausted) {
                    val newInstances = B1.sampleBinaryInstancesEDIS()
                    // 只检查新采样的实例
                    val newMatchCount = newInstances.count { it in headInstances }
                    S_H1_size += newMatchCount
                    if (shouldDebug)
                    println("\t[Thread-$threadId] ${B1} sampling round ${B1.samplingRound}: " +
                            "new=${newInstances.size}, total=${B1.instances.size}, " +
                            "S_H1=$S_H1_size, exhausted=${B1.samplingExhausted}")
                }
            }
        }
        if (shouldDebug)
        println("[Thread-$threadId] ${B1} total sampling rounds ${B1.samplingRound}: " +
                    "total=${B1.instances.size}, S_H1=$S_H1_size, exhausted=${B1.samplingExhausted}")
        
        if (S_H1_size < Settings.MIN_SUPP) {
            return  // Does not meet minimum support even after sampling
        }
        
        for (j in (i + 1) until bodyList.size) {
                // Check thread interruption
                if (Thread.currentThread().isInterrupted) {
                    println("Thread interrupted, exiting processBinaryHeadAtom for $headAtom (index: $i)")
                    return
                }
                
                val (B2, ruleId2, metric2) = bodyList[j]
                if (metric1.surprisal + metric2.surprisal >= Settings.MAX_SURPRISAL) {
                    continue
                }
                
                pairCount++
                
                var S_12_size = 0
                var S_H12_size = 0
                
                // 先检查 B1 已有的 instances
                for (e in B1.instances) {
                    if (B2.hasBinaryInstance(e)) {
                        S_12_size++
                        if (e in headInstances) {
                            S_H12_size++
                        }
                    }
                }
                
                val initialS12 = S_12_size
                val initialSH12 = S_H12_size
                
                // Dynamic sampling loop for B1
                // 只对非L1原子进行采样
                while (S_H12_size < Settings.MIN_SUPP && !B1.isL1Atom && !B1.samplingExhausted) {
                    val newInstances = B1.sampleBinaryInstancesEDIS()
                    
                    var newS12 = 0
                    var newSH12 = 0
                    
                    // 只检查新采样的实例
                    for (e in newInstances) {
                        if (B2.hasBinaryInstance(e)) {
                            S_12_size++
                            newS12++
                            if (e in headInstances) {
                                S_H12_size++
                                newSH12++
                            }
                        }
                    }
                    // if (shouldDebug)
                    // println("\t[Thread-$threadId] Pair($i,$j) sampling round ${B1.samplingRound}: " +
                    //         "newInstances=${newInstances.size}, newS12=$newS12, newSH12=$newSH12, S_12=$S_12_size, S_H12=$S_H12_size, " + "exhausted=${B1.samplingExhausted}")
                }
                // if (shouldDebug)
                // println("[Thread-$threadId] Pair($i,$j) total sampling rounds ${B1.samplingRound}: " +
                //             "S_12=${S_12_size}, S_H12=${S_H12_size}, exhausted=${B1.samplingExhausted}")
                
                if (S_H12_size < Settings.MIN_SUPP) {
                    continue  // Does not meet minimum support
                }
                
                // Create new metric with bodySize = S_12_size
                val metric = Metric(
                    support = S_H12_size.toDouble(),
                    headSize = headInstances.size,
                    bodySize = S_12_size
                )
                
                storeDependency(ruleId1, ruleId2, metric, metric1, metric2, binaryPositiveLift, binaryNegativeLift)
        }
        // 使用新指标更新B1
        val S_H1 = B1.instances.count { it in headInstances }
        val newMetric = Metric(
            support = S_H1.toDouble(),
            headSize = headInstances.size,
            bodySize = B1.instances.size
        )
        ID2metric[ruleId1] = newMetric
        if (shouldDebug) {
            println("[Thread-$threadId] processBinaryHeadAtom completed: $headAtom, " +
                    "checked $pairCount pairs")
        }
    }
    
    /**
     * Process single unary headAtom, perform pairwise combination of bodyAtoms
     * Uses exact set operations on unary instances
     */
    private fun processUnaryHeadAtom(
        headAtom: DepAtom,
        bodyList: List<Triple<DepAtom, Int, Metric>>,
        binaryBodyList: List<Triple<DepAtom, Int, Metric>>,
        i: Int
    ) {
        if (bodyList.isEmpty()) return
        if (i !in bodyList.indices) return
        
        var pairCount = 0
        // Pairwise combination: only combine (i, j) where i < j to avoid duplicates
        val (B1, ruleId1, metric1) = bodyList[i]
        val B1_instances = B1.getUnaryInstances()
        val headInstances = headAtom.getUnaryInstances()
        val S_H1 = B1_instances.intersect(headInstances)
        if (S_H1.size < Settings.MIN_SUPP) {
            return  // Does not meet minimum support
        }

        // Unary-Binary dependency
        for (k in binaryBodyList.indices) {
            if (Thread.currentThread().isInterrupted) {
                println("Thread interrupted, exiting processUnaryHeadAtom for $headAtom (index: $i)")
                return
            }

            val (B2, ruleId2, metric2) = binaryBodyList[k]
            if (metric1.surprisal + metric2.surprisal >= Settings.MAX_SURPRISAL) {
                continue
            }

            var S_H12_size = 0
            for (h in S_H1) {
                val instance = if (B1.isInverseRelation)  packBinaryInstance(B1.entityId, h)
                else packBinaryInstance(h, B1.entityId)
                if (B2.hasBinaryInstance(instance)) S_H12_size++
            }

            if (S_H12_size < Settings.MIN_SUPP) {
                continue  // Does not meet minimum support
            }

            var S_12_size = 0
            for (h in B1_instances) {
                val instance = if (B1.isInverseRelation) packBinaryInstance(B1.entityId, h)
                else packBinaryInstance(h, B1.entityId)
                if (B2.hasBinaryInstance(instance)) S_12_size++
            }

            val metric = Metric(
                support = S_H12_size.toDouble(),
                headSize = headInstances.size,
                bodySize = S_12_size
            )

            storeDependency(ruleId1, ruleId2, metric, metric1, metric2, mixPositiveLift, mixNegativeLift)
        }

        for (j in (i + 1) until bodyList.size) {
            // === 响应线程中断 ===
            if (Thread.currentThread().isInterrupted) {
                println("Thread interrupted, exiting processUnaryHeadAtom for $headAtom (index: $i)")
                return
            }

            val (B2, ruleId2, metric2) = bodyList[j]
            if (metric1.surprisal + metric2.surprisal >= Settings.MAX_SURPRISAL) {
                println("Skipping pair with high combined surprisal: ${metric1.surprisal} + ${metric2.surprisal}")
                continue
            }

            pairCount++

            val B2_instances = B2.getUnaryInstances()
            var S_H12_size = S_H1.intersect(B2_instances).size
            
            if (S_H12_size < Settings.MIN_SUPP) {
                continue  // Does not meet minimum support
            }
            
            // Calculate common evidence: intersection of two bodyAtom instances
            val S_12_size = B1_instances.intersect(B2_instances).size

            // Create new metric with bodySize = |S_12|
            val metric = Metric(
                support = S_H12_size.toDouble(),
                headSize = headInstances.size,
                bodySize = S_12_size
            )

            // Calculate lift
            storeDependency(ruleId1, ruleId2, metric, metric1, metric2, unaryPositiveLift, unaryNegativeLift)
        }
    }
    
    /**
     * Save metric map to JSON file - streaming output to avoid memory overflow
     * @param metricMap The metric map to save (H2B2metric or H2F2metric)
     * @param outputPath The output JSON file path
     * @param appendMode Whether to append to existing rules file (true for H2F, false for H2B)
     * @param isFormulaMap Whether the body type is DepFormula (true) or DepAtom (false)
     */
    private fun saveMetricToJson(
        metricMap: ConcurrentHashMap<DepAtom, ConcurrentHashMap<DepAtom, Int>>,
        outputPath: String,
        appendMode: Boolean,
        isFormulaMap: Boolean
    ) {
        val outputFile = File(outputPath)
        val outputRule = File(Settings.PATH_RULES_TXT)
        outputFile.parentFile?.mkdirs()
        outputRule.parentFile?.mkdirs()
        
        val metricType = if (isFormulaMap) "H2F2metric" else "H2B2metric"
        println("Saving $metricType to ${outputFile.absolutePath}...")
        
        java.io.BufferedWriter(java.io.FileWriter(outputFile)).use { writer ->
            java.io.BufferedWriter(java.io.FileWriter(outputRule, appendMode)).use { ruleWriter ->
                writer.write("{\n")
                val atomEntries = metricMap.entries.toList()

                atomEntries.forEachIndexed { atomIndex, (atom, bodyMap) ->
                    val headAtomString = atom.toString().replace("\"", "\\\"").replace("\n", "\\n")
                    writer.write("  \"$headAtomString\": {\n")

                    val bodyEntries = bodyMap.entries.toList()
                        .mapNotNull { entry ->
                            val metric = ID2metric[entry.value] ?: return@mapNotNull null
                            Triple(entry.key, entry.value, metric)
                        }
                        .sortedByDescending { it.third.confidence }
                    
                    bodyEntries.forEachIndexed { bodyIndex, (body, ruleId, metric) ->
                        val bodyString = body.toString().replace("\"", "\\\"").replace("\n", "\\n")
                        writer.write("    \"$bodyString\": $metric")
                        if (bodyIndex < bodyEntries.size - 1) writer.write(",")
                        writer.write("\n")
                        
                        // Get rule string based on body type
                        val bodyRuleString = body.getRuleString()
                        
                        // Write rule to text file with lift info for formulas
                        val liftInfo = if (isFormulaMap) metric.lift else metric.confidence
                        val ruleLine = "${metric.bodySize}\t${metric.support.toInt()}\t${format5(liftInfo)}\t${atom.getRuleString()} <= $bodyRuleString"
                        ruleWriter.write(ruleLine)
                        ruleWriter.write("\n")
                    }

                    writer.write("  }")
                    if (atomIndex < atomEntries.size - 1) writer.write(",")
                    writer.write("\n")

                    if ((atomIndex+1) % 1000 == 0) {
                        writer.flush()
                        ruleWriter.flush()
                        println("[save${metricType}ToJson] Processed ${atomIndex + 1}/${atomEntries.size} head atoms...")
                    }
                }
                writer.write("}\n")
            }
        }

        println("Successfully saved $metricType to ${outputFile.absolutePath}")
        println("Successfully saved rules to ${outputRule.absolutePath}")
        println("Total head atoms: ${metricMap.size}")
        println("Total body entries: ${metricMap.values.sumOf { it.size }}")
    }

    /**
     * Save dependency metrics to file
     * Format: bodySize\tsupp\tlift\tID1\tID2
     *
     * Additional split files:
     * - synerge.txt: positive lift dependencies (lift > 0)
     * - redundancy.txt: negative/zero lift dependencies (lift <= 0)
     */
    private fun saveDependencyToFile(outputPath: String) {
        val outputFile = File(outputPath)
        outputFile.parentFile?.mkdirs()
        println("Saving depAdj2metric to ${outputFile.absolutePath}...")

        val synergeFile = File(outputFile.parentFile, "synergy.txt")
        val redundancyFile = File(outputFile.parentFile, "redundancy.txt")

        val jsonOutputPath = outputPath.replace(".txt", ".json")
        val jsonOutputFile = File(jsonOutputPath)
        jsonOutputFile.parentFile?.mkdirs()

        val sortedEntries = depAdj2metric.entries
            .flatMap { (id1, inner) -> inner.entries.map { id1 to it } }
            .sortedByDescending { it.second.value.confidence }

        // PrintWriter(outputFile).use { writer ->
        //     sortedEntries.forEach { (id1, entry) ->
        //         val id2 = entry.key
        //         val metric = entry.value
        //         val line = "${metric.bodySize}\t${metric.support.toInt()}\t${format5(metric.lift)}\t$id1\t$id2"
        //         writer.println(line)
        //     }
        // }

        var positiveDeps = 0
        var negativeDeps = 0
        PrintWriter(synergeFile).use { synergeWriter ->
            PrintWriter(redundancyFile).use { redundancyWriter ->
                sortedEntries.forEach { (id1, entry) ->
                    val id2 = entry.key
                    val metric = entry.value
                    val line = "${metric.bodySize}\t${metric.support.toInt()}\t${format5(metric.lift)}\t$id1\t$id2"
                    if (metric.lift > 0) {
                        synergeWriter.println(line)
                        positiveDeps++
                    } else {
                        redundancyWriter.println(line)
                        negativeDeps++
                    }
                }
            }
        }

        PrintWriter(jsonOutputFile).use { writer ->
            writer.println("{")
            val entries = depAdj2metric.entries.toList()
            entries.forEachIndexed { idx, (id1, innerMap) ->
                val inner = innerMap.entries
                    .joinToString(", ") { (id2, metric) ->
                        "\"$id2\": ${format5(metric.lift)}"
                    }
                val comma = if (idx < entries.size - 1) "," else ""
                writer.println("  \"$id1\": { $inner }$comma")
            }
            writer.println("}")
        }

        println("Successfully saved depAdj2metric to ${outputFile.absolutePath}")
        println("Successfully saved positive dependencies to ${synergeFile.absolutePath}")
        println("Successfully saved negative/zero dependencies to ${redundancyFile.absolutePath}")
        println("Successfully saved dependency json to ${jsonOutputFile.absolutePath}")
        val totalDeps = depAdj2metric.values.sumOf { it.size }
        println("Positive dependencies (synerge): $positiveDeps")
        println("Negative/zero dependencies (redundancy): $negativeDeps")
        println("Total dependency entries: $totalDeps")
    }

    /**
     * Print statistics about H2B2metric and rules
     */
    fun printStatistics() {
        println("\n=== Statistics ===")
        println("=== Metric Statistics ===")
        
        val totalHeads = H2B2ID.size
        val totalBodyAtoms = H2B2ID.values.sumOf { it.size }
        val totalFormulas = depAdj2metric.values.sumOf { it.size }
        val avgBodyPerHead = if (totalHeads > 0) totalBodyAtoms.toDouble() / totalHeads else 0.0
        
        println("Total head atoms: $totalHeads")
        println("Total H2B rules: $totalBodyAtoms")
        println("Total H2F rules: $totalFormulas")
        println("Average body atoms per head: ${"%.2f".format(avgBodyPerHead)}")
        
        // Breakdown by atom type
        var binaryHeads = 0
        var loopHeads = 0
        var constantHeads = 0
        
        for (head in H2B2ID.keys) {
            when {
                head.entityId == IdManager.getYId() -> binaryHeads++
                head.entityId == IdManager.getXId() -> loopHeads++
                head.entityId == 0 -> throw IllegalStateException("Head atom cannot be existence (*)")
                else -> constantHeads++
            }
        }
        
        println("\nHead atom types:")
        println("  Binary (X,Y): $binaryHeads")
        println("  Loop (X,X): $loopHeads")
        println("  Constant (X,e): $constantHeads")

        // Print lift statistics for composition phase
        println("\nComposition Phase - Lift Statistics:")
        println("-".repeat(60))
        println("Type     Positive Lift    Negative Lift    Total")
        val unaryTotal = unaryPositiveLift.get() + unaryNegativeLift.get()
        val binaryTotal = binaryPositiveLift.get() + binaryNegativeLift.get()
        val mixTotal = mixPositiveLift.get() + mixNegativeLift.get()
        println("Unary    ${unaryPositiveLift.get().toString().padStart(13)}    ${unaryNegativeLift.get().toString().padStart(13)}    ${unaryTotal.toString().padStart(8)}")
        println("Binary   ${binaryPositiveLift.get().toString().padStart(13)}    ${binaryNegativeLift.get().toString().padStart(13)}    ${binaryTotal.toString().padStart(8)}")
        println("Mix      ${mixPositiveLift.get().toString().padStart(13)}    ${mixNegativeLift.get().toString().padStart(13)}    ${mixTotal.toString().padStart(8)}")
        println("Total    ${(unaryPositiveLift.get() + binaryPositiveLift.get() + mixPositiveLift.get()).toString().padStart(13)}    ${(unaryNegativeLift.get() + binaryNegativeLift.get() + mixNegativeLift.get()).toString().padStart(13)}    ${(unaryTotal + binaryTotal + mixTotal).toString().padStart(8)}")
    }
}
