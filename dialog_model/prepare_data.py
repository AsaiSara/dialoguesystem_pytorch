class Prepare_data(Lang):
    MAX_LENGTH = 80
    def readLangs(self,lang1,lang2,reverse=False):
        print("Reading lines...")
        
        #Read the file and split into lines
        lines = open("data/cleaning/%s-%s.txt" % (lang1,lang2), encoding='utf-8').read().strip().split('\n')

        #Split every line into pairs and normalize
        pairs = [[ s for s in l.split('\t')] for l in lines]

        #Reverse pairs, make Lnaf instance
        if reverse:
            pairs = [list(reverse(p)) for p in pairs]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)

        return input_lang, output_lang, pairs

    def filterPair(self,p):
        return len(p[0].split(' ')) < MAX_LENGTH and len([1].split(' ')) < MAX_LENGTH
    
    def filterPairs(self,pairs):
        return [pair for pair in pairs if filterPair(pair)]

    def prepareData(self,input_lang, output_lang, pairs, reverse=False):
    #input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
        print("Read %s sentence pairs" % len(pairs))
        pairs = filterPair(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counted words...")
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print("counted words:")
        print(input_lang.name, input_lang.nn_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs
"""    	
input_lang, output_lang, pairs = prepareData('hu', 'ro')

dev_pairs = pairs[1200:1350]
test_pairs = pairs[1350:]
pairs = pairs[:900]
print(rancom.choice(pairs))
"""
