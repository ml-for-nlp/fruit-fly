import sys
import utils
import MEN
import numpy as np

'''Parameter input'''

if len(sys.argv) < 5 or sys.argv[1] not in ("bnc","wiki"):
    print("\nUSAGE: python3 projection.py bnc|wiki [num-kc] [size-proj] [percent-hash]\n\
    - num-kc: the number of Kenyon cells\n\
    - size-proj: how many projection neurons are used for each projection\n\
    - percent-hash: how much of the Kenyon layer to keep in the final hash.\n")
    sys.exit() 

if sys.argv[1] == "bnc":
    data = "data/BNC-MEN.dm"
    column_labels = "data/BNC-MEN.cols"
    MEN_annot = "data/MEN_dataset_lemma_form_full"
else:
    data = "data/wiki_all.dm"
    column_labels = "data/wiki_all.cols"
    MEN_annot = "data/MEN_dataset_natural_form_full"

english_space = utils.readDM(data)
i_to_cols, cols_to_i = utils.readCols(column_labels)

PN_size = len(i_to_cols)
KC_size = int(sys.argv[2])
proj_size = int(sys.argv[3])
percent_hash = int(sys.argv[4])
print("SIZES PN LAYER:",PN_size,"KC LAYER:",KC_size)
print("SIZE OF PROJECTIONS:",proj_size)
print("SIZE OF FINAL HASH:",percent_hash,"%")

projection_layer = np.zeros(PN_size)
kenyon_layer = np.zeros(KC_size)
projection_functions = []


'''Create random projections'''
print("Creating",KC_size,"random projections...")
projection_functions = {}
for cell in range(KC_size):
    activated_pns = np.random.randint(PN_size, size=proj_size)
    projection_functions[cell] = activated_pns

def show_projections(word,hashed_kenyon):
    important_words = {}
    for i in range(len(hashed_kenyon)):
        if hashed_kenyon[i] == 1:
            activated_pns = projection_functions[i]
            #print(word,[i_to_cols[pn] for pn in activated_pns])
            for pn in activated_pns:
                w = i_to_cols[pn]
                if w in important_words:
                    important_words[w]+=1
                else:
                    important_words[w]=1
    print(word,"BEST PNS", sorted(important_words, key=important_words.get, reverse=True)[:proj_size])

def projection(projection_layer):
    kenyon_layer = np.zeros(KC_size)
    for cell in range(KC_size):
        activated_pns = projection_functions[cell]
        for pn in activated_pns:
            kenyon_layer[cell]+=projection_layer[pn]
    return kenyon_layer

def hash_kenyon(kenyon_layer):
    #print(kenyon_layer[:100])
    kenyon_activations = np.zeros(KC_size)
    top = int(percent_hash * KC_size / 100)
    activated_kcs = np.argpartition(kenyon_layer, -top)[-top:]
    for cell in activated_kcs:
        kenyon_activations[cell] = 1
    return kenyon_activations

def hash_input(word):
    projection_layer = english_space[word]
    kenyon_layer = projection(projection_layer)
    hashed_kenyon = hash_kenyon(kenyon_layer)
    if len(sys.argv) == 6 and sys.argv[5] == "-v":
        show_projections(word,hashed_kenyon)
    return hashed_kenyon

english_space_hashed = {}

for w in english_space:
    hw = hash_input(w)
    english_space_hashed[w]=hw

#print(utils.neighbours(english_space,sys.argv[1],10))
#print(utils.neighbours(english_space_hashed,sys.argv[1],10))

sp,count = MEN.compute_men_spearman(english_space,MEN_annot)
print ("SPEARMAN BEFORE FLYING:",sp, "(calculated over",count,"items.)")
sp,count = MEN.compute_men_spearman(english_space_hashed,MEN_annot)
print ("SPEARMAN AFTER FLYING:",sp, "(calculated over",count,"items.)")
