import random
import algo.textprocessing as tp

class EndingRefiner:
    def refine(self, context_document, ending_document):
        raise NotImplementedError

class NopEndingRefiner(EndingRefiner):
    def refine(self, context_document, ending_document):
        return tp.get_sentence_from_conll(ending_document)

class ProperNounRefiner(EndingRefiner):
    def refine(self, context_document, ending_document):
        refined_ending_document = list(ending_document)

        context_nnp_sg = set()
        context_nnp_pl = set()
        for tup in context_document:
            pos = tup[2]
            if pos == "NNP":
                context_nnp_sg.add(tup)
            elif pos == "NNPS":
                context_nnp_pl.add(tup)

        for i, tup in enumerate(refined_ending_document):
            pos = tup[2]
            if pos == "NNP" and context_nnp_sg:
                refined_ending_document[i] = random.choice(tuple(context_nnp_sg))
            elif pos == "NNPS" and context_nnp_pl:
                refined_ending_document[i] = random.choice(tuple(context_nnp_pl))

        return tp.get_sentence_from_conll(refined_ending_document)