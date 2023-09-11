import pandas as pd
import numpy as np
import networkx as nx

# Set paths

path_snomed_concept = "CONCEPT.csv"
path_snomed_relationship = "CONCEPT_RELATIONSHIP.csv"

path_12sl = "../12sl_statements.csv"
path_12sl_mapped_to_snomed = "../12slv23ToSNOMED.csv"

path_ptbxl = "../ptbxl_statements.csv"
path_ptbxl_mapped_to_snomed = "../ptbxlToSNOMED.csv"


# Utility functions

def flatten(l):
    return [item for sublist in l for item in sublist]

def flatten_unique(l):
    return list(set(flatten(l)))

def convert_to_int(x):
    try:
        xi = int(x)
    except:
        xi = -1
    return xi

def get_parents(concept_id, df_relationship):
    lst = list(df_relationship[(df_relationship.concept_id_1==concept_id) & (df_relationship.relationship_id == "Is a")]["concept_id_2"])
    return [x for x in lst if x!=concept_id]

def get_name_from_id(concept_id, df_concept,include_id=False):
    selection = df_concept[df_concept.concept_id ==concept_id]
    if(len(selection)>0):
        name = selection.iloc[0]["concept_name"]
        if(include_id):
            name = name + "["+str(concept_id)+"]"
    else:
        name = "invalid id ["+str(concept_id)+"]"
        print("Invalid ID:",concept_id)
    return name

def get_id_from_code(concept_code, df_concept):
    selection = df_concept[df_concept.concept_code ==concept_code]
    if(len(selection)>0):
        return selection.iloc[0]["concept_id"]
    else:
        print("Invalid code:",concept_code)
        return -1

def get_name_from_code(concept_code, df_concept,include_id=False):
    idx = get_id_from_code(concept_code,df_concept)
    return get_name_from_id(idx,df_concept,include_id)

def populate_graph(lst,G=None,lst_processed=[],include_id=False,only_id=False):
    if(G is None):
        G = nx.DiGraph()
    if(len(lst)==0):
        return G
    else:
        tag = lst[0] if only_id else get_name_from_id(lst[0], df_concept,include_id=include_id)

        for p in get_parents(lst[0], df_relationship):
            tagp = p if only_id else get_name_from_id(p, df_concept, include_id=include_id)
            if(not p in lst_processed):
                lst.append(p)
            G.add_edge(tagp,tag)
            
        lst0=lst[0]
        lst.pop(0)
        return populate_graph(list(set(lst)),G,lst_processed+[lst0], include_id=include_id, only_id=only_id)

def get_uppropagated_labels(key_lst,G,exclude_snomed_id_lst=[]):
    result=[]
    for key in key_lst:
        #assumes DAG
        for s in [key]:
            tmp=[l for l in list(nx.ancestors(G,s)) if not l in exclude_snomed_id_lst]
            if not s in exclude_snomed_id_lst:
                tmp+=[s]
            result+=tmp
    return list(set(result))

def reformat_ge(x,remove_brackets=True):
    if(remove_brackets):
        x= " ".join(x).replace(" COMMA "," ").replace(" LPAREN "," ").replace(" RPAREN "," ")
    else:
        x= " ".join(x).replace(" COMMA ",";,;").replace(" LPAREN ",";(;").replace(" RPAREN",";)")#human readable
    x= x.replace(" $SWITH ",";$SWITH;").replace(" $SOR ",";$SOR;").replace(" $SAND ",";$SAND;")#with, or, and
    for s in ["AC","AU","OLD","NEW"]:#infarction combine with previous
        x= x.replace(" "+s,";"+s)
    for s in ["BLKED","ACCEL","PO","CRO"]:#combine with next
        x= x.replace(s+" ",s+";")
    for s in ["FAV","SPR","MBZI","MBZII","SAV","CHB","VAVB",
              "AVDIS","W2T1","W3T1","W4T1","RVR","SVR","CJP",
              "IRREG","ABER","PROAV","CSEC","BIGEM","JESC","VESC",
             "$SRETC","SAR","MSAR","RVE+","QRSW","2ST","QRSW-2ST","MAFB"]:#with... combine with previous
        x= x.replace(" "+s,";"+s)
    x=x.replace(" OCC ",";OCC;")
    x=x.replace(" FREQ ",";FREQ;")
    x=x.replace("ST& ","ST&;")
    return x

def minimal_extension(x,with_uncert=True):#only bind AC and AU
    if(with_uncert):
        x=x.replace("PO; ","PO;").replace("CRO; ","CRO;")
    else:
        x=x.replace("PO;","").replace("CRO;","")
    #remove non-informative tokens
    for l in ["$SWITH;","$SOR;","$SAND;","OCC;","FREQ;"]:
        x = x.replace(l,"")
    x= x.replace("LVH ","")
    for l in ["CSEC","RAVL","SOKOLYON","CORNPROD","ROMESTES","QRSV","LVH3","LPAREN","RPAREN"]:
        x = x.replace(" "+l,"")
        x = x.replace(l+" ","")
        
    #fix issues with ACCEL and swap MAFB and AU first
    return x.replace(";ACCEL"," ACCEL").replace("MAFB;AU","AU;MAFB").replace(";AU","xxxAU").replace(";AC","xxxAC").replace("PO;","POxxx").replace("CRO;","CROxxx").replace(";"," ").replace("xxx",";").strip()

def apply_snomed_mapping_ge(x):
    if(x.startswith("PO;")):
        cert=uncertainty_mapping["PO"]
        statement=x[3:]
    elif(x.startswith("CRO;")):
        cert=uncertainty_mapping["CRO"]
        statement=x[4:]
    else:
        cert=100.
        statement=x
    x_mapped=ge_to_snomed[statement]
    return [(x[0],min(x[1],cert)) for x in x_mapped] #in doubt take the less certain value

def map_certainty_statements_ext(x):
    if(x.startswith("PO;")):
        cert=uncertainty_mapping["PO"]
        statement=x[3:]
    elif(x.startswith("CRO;")):
        cert=uncertainty_mapping["CRO"]
        statement=x[4:]
    else:
        cert=100.
        statement=x
    return (statement,cert)

def map_infarction_stadium(stadium1, stadium2):
    if(stadium1 in ['Stadium I', 'Stadium I-II']):
        return 1
    if(stadium2 in ['Stadium I', 'Stadium I-II']):
        return 1
    if(stadium1 == 'Stadium II'):
        return 2
    if(stadium2 == 'Stadium II'):
        return 2
    if(stadium1 in ['Stadium III', 'Stadium II-III']):
        return 3
    if(stadium2 in ['Stadium III', 'Stadium II-III']):
        return 3
    return 0

def map_mi_labels(labels, infarction_stadium):
    mi_labels = ["IMI","ASMI","ILMI","AMI","ALMI","INJAS","LMI","INJAL","IPLMI","IPMI","INJIN","PMI","INJLA","INJIL"]
    if(infarction_stadium==1):#add more specific acute labels
        return labels + [(l[0]+"_AC",l[1]) for l in labels if l[0] in mi_labels]
    elif(infarction_stadium==2):#add more specific old labels
        return labels + [(l[0]+"_OLD",l[1]) for l in labels if l[0] in mi_labels]
    return labels

def apply_snomed_mapping_ptbxl(x):
    return [(a,x[1]) for a in ptbxl_to_snomed[x[0]]]

def apply_uppropagation_dict(lst):
    lst_ext = [([l[0]]+uppropagation_dict[l[0]],l[1]) for l in lst]
    lst_ext = flatten([[(x,l[1]) for x in l[0]] for l in lst_ext])
    #remove duplicate entries (in doubt take the more certain value)
    labels = np.array([l[0] for l in lst_ext])
    confidences = np.array([l[1] for l in lst_ext])
    output=[]
    for l in np.unique(labels):
        output.append((l,max(confidences[np.where(labels==l)[0]])))
    return output
    

# Parsing the SNOMED label tree

df_concept=pd.read_csv(path_snomed_concept,sep='\t')
df_concept.concept_id=df_concept.concept_id.apply(lambda x: convert_to_int(x))
df_concept.concept_code=df_concept.concept_code.apply(lambda x: convert_to_int(x))
df_relationship=pd.read_csv(path_snomed_relationship,sep='\t')

# Load 12SL labels and mapping

# Note (according to PTB-XL docs): cannot rule out (CRO) weight 0.15, consider weight 0.35, possible (PO) weight 0.5, probable weight 0.5



df_labels = pd.read_csv(path_12sl)

df_labels.columns

df_labels["statements"]=df_labels["statements"].apply(lambda x: eval(x))

df_labels["statements_cat"]=df_labels["statements"].apply(lambda x: reformat_ge(x))
df_labels["statements_ext"]=df_labels["statements_cat"].apply(lambda x: minimal_extension(x,with_uncert=True).split(" "))

#finally split into lists again
df_labels["statements_cat"]=df_labels["statements_cat"].apply(lambda x:x.split(" "))

uncertainty_mapping = {np.nan:100., "consider":35., "possible":50., "probable":50., "probably":50., "PO":50., "CRO":15.}

df_ge_snomed = pd.read_csv(path_12sl_mapped_to_snomed)

ge_to_snomed = {}

for _,row in df_ge_snomed.iterrows():
    if(not type(row["Acronym"])==str):
        continue
    tmp = []
    if(not np.isnan(row["id1"])):
        tmp.append((int(row["id1"]),uncertainty_mapping[row["qualifier1"]]))
    if(not np.isnan(row["id2"])):
        tmp.append((int(row["id2"]),uncertainty_mapping[row["qualifier2"]]))
    if(not np.isnan(row["id3"])):
        tmp.append((int(row["id3"]),uncertainty_mapping[row["qualifier3"]]))
    if(not np.isnan(row["id4"])):
        tmp.append((int(row["id4"]),uncertainty_mapping[row["qualifier4"]]))
    if(not np.isnan(row["id5"])):
        tmp.append((int(row["id5"]),uncertainty_mapping[row["qualifier5"]]))
    ge_to_snomed[row["Acronym"]]=tmp
    

df_labels["statements_ext_snomed"]=df_labels["statements_ext"].apply(lambda x:flatten([apply_snomed_mapping_ge(l) for l in x]))

df_labels["statements_ext"]=df_labels["statements_ext"].apply(lambda x: [map_certainty_statements_ext(y) for y in x])

# Load PTB-XL labels and mapping

df_ptbxl = pd.read_csv(path_ptbxl)

df_ptbxl_snomed = pd.read_csv(path_ptbxl_mapped_to_snomed)

ptbxl_to_snomed = {}

for _,row in df_ptbxl_snomed.iterrows():
    if(not type(row["Acronym"])==str):
        continue
    tmp = []
    if(not np.isnan(row["id1"])):
        tmp.append(int(row["id1"]))
    if(not np.isnan(row["id2"])):
        tmp.append(int(row["id2"]))
    if(not np.isnan(row["id3"])):
        tmp.append(int(row["id3"]))
    if(not np.isnan(row["id4"])):
        tmp.append(int(row["id4"]))
    ptbxl_to_snomed[row["Acronym"]]=tmp

df_ptbxl["scp_codes"]=df_ptbxl.scp_codes.apply(lambda x: eval(x))
df_ptbxl["scp_codes_ext"]=df_ptbxl.scp_codes_ext.apply(lambda x: eval(x))

df_ptbxl["scp_codes_ext_snomed"]=df_ptbxl["scp_codes_ext"].apply(lambda x:flatten([apply_snomed_mapping_ptbxl(l) for l in x]))

## Populate SNOMED label tree

ge_snomed_ids = flatten_unique([[l[0] for l in x] for x in df_labels.statements_ext_snomed])
ptbxl_snomed_ids = flatten_unique([[l[0] for l in x] for x in df_ptbxl.scp_codes_ext_snomed])
all_snomed_ids = list(np.unique(ge_snomed_ids+ptbxl_snomed_ids))

G = populate_graph(all_snomed_ids, only_id=True)

all_snomed_ids_uppropagated=flatten_unique([get_uppropagated_labels([i],G) for i in all_snomed_ids])

node_description = []
for i in all_snomed_ids_uppropagated:
    node_description.append({"snomed_id":i, "description":get_name_from_id(i, df_concept, include_id=False), "ancestors":[x for x in get_uppropagated_labels([i],G) if x!=i]})
df_snomed_description=pd.DataFrame(node_description)

#save new snomed description table
df_snomed_description.to_csv("snomed_description_new.csv",index=False)

uppropagation_dict={}

for _,row in df_snomed_description.iterrows():
    uppropagation_dict[row["snomed_id"]]=row["ancestors"]

## replace snomed labels by uppropagated labels

df_labels["statements_ext_snomed"]=df_labels["statements_ext_snomed"].apply(lambda x: apply_uppropagation_dict(x))

df_ptbxl["scp_codes_ext_snomed"]=df_ptbxl["scp_codes_ext_snomed"].apply(lambda x: apply_uppropagation_dict(x))
df_ptbxl = df_ptbxl[["ecg_id","scp_codes","scp_codes_ext","scp_codes_ext_snomed"]]

#save new mapped labels
df_labels.to_csv("./output/labels/12sl_statements_new.csv",index=False)
df_ptbxl.to_csv("./output/labels/ptbxl_statements_new.csv",index=False)
