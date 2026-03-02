import argparse
import pickle
import os
from collections import defaultdict
import re
from tqdm import tqdm
pattern_text = r'(?P<relation>.+)\((?P<x>.+),(?P<y>.+)\)'
pattern = re.compile(pattern_text)

def parse_rule(rule):
    if len(rule.split(" <= ")) != 2:
        return None
    head, body = rule.split(" <= ")
    atoms = body.split(", ")        
    head_match = pattern.match(head).groups()
    body_match = [pattern.match(atom).groups() for atom in atoms]
    return head_match, body_match

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Filters explanations')
    parser.add_argument('-e','--explanation', help='Folder containing processed explanations', default="/fs/scratch/rng_cr_bcai_dl_students/ots2rng/fb15k-237/expl/explanations-processed/")
    parser.add_argument('-o','--output', help='Folder where the filtereed explanations are written', default="/fs/scratch/rng_cr_bcai_dl_students/ots2rng/fb15k-237/expl-filt/explanations-processed/")
    # parser.add_argument('-e','--explanation', help='Folder containing processed explanations', default="/home/ots2rng/kge/data/codex-m/expl/explanations-processed/")
    # parser.add_argument('-o','--output', help='Folder where the filtereed explanations are written', default="/home/ots2rng/kge/data/codex-m/expl-filt/explanations-processed/")
    args = vars(parser.parse_args())
    
    
    if not os.path.exists(args["output"]):
        os.makedirs(args["output"])

    rule_map = pickle.load(open(args["explanation"] + "rule_map.pkl", "rb"))
    rule_features = pickle.load(open(args["explanation"] + "rule_features.pkl", "rb"))
    
    map_old_new = {}
    rule_features_new = {}
    rule_map_new = defaultdict(list)
    filtered_rules = []
    
    print("Generating new filtered rule_map and features...")
    curr_id = 0
    for relation in rule_map:
        rules = rule_map[relation]
        for rule in rules:
            rule_feature = rule_features[rule]
            
            head, body = parse_rule(rule_feature[2])
            if len(body) == 1:
                # index 0 = r, index 1 = h, index 2 = t
                if (body[0][1] == "X" and body[0][2] == "Y") or (body[0][1] == "Y" and body[0][2] == "X"):
                    # cyclic rule len 1
                    filtered_rules.append(rule_feature[2] + "\n")
                    continue
                elif head[1] != "X" and (head[1] == body[0][1] or head[1] == body[0][2]):
                    # h(c,Y) <= r(c,Y) or
                    # h(c,Y) <= r(Y,c)
                    filtered_rules.append(rule_feature[2]+ "\n")
                    continue
                elif head[2] != "Y" and (head[2] == body[0][1] or head[2] == body[0][2]):  
                    # h(X,c) <= r(c,X) or
                    # h(X,c) <= r(X,c)
                    filtered_rules.append(rule_feature[2]+ "\n")
                    continue
            
            # if rule_feature[2].count(', ') > 0:
            map_old_new[curr_id] = rule
            rule_features_new[curr_id] = rule_feature
            rule_map_new[relation].append(curr_id)
            curr_id += 1
            
            
    print("Old rulelen:", len(rule_features))
    print("New rulelen", len(rule_features_new))
    
    print("Writing new rule_map and features...")
    with open(args["output"] + "rule_map.pkl", 'wb') as f:
        pickle.dump(rule_map_new, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(args["output"] + "rule_features.pkl", 'wb') as f:
        pickle.dump(rule_features_new, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(args["output"] + "filtered.txt", 'w') as f:
        f.writelines(filtered_rules)
    
    for processed_name in ["processed_sp_train", "processed_po_train", "processed_sp_test", "processed_po_test", "processed_sp_valid", "processed_po_valid"]:
        print("Filter " + processed_name + " ...")
        
        processed = pickle.load(open(args["explanation"] + f"{processed_name}.pkl", "rb"))
        for key in tqdm(processed):
            rules_new = []
            for rules_c in processed[key]["rules"]:
                rules_c_new = []
                for rule in rules_c:
                    if rule in map_old_new:
                        rules_c_new.append(map_old_new[rule])
                rules_new.append(rules_c_new)
            processed[key]["rules"] = rules_new
        assert len(processed[key]["rules"]) == len(processed[key]["candidates"])
        with open(args["output"] + f"{processed_name}.pkl", 'wb') as f:
            pickle.dump(processed, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done")
        
        
            
        
        
        
        
        
        