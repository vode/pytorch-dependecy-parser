from optparse import OptionParser
import pickle
import util
import model
import os
import os.path
import time
from model import Train

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file",
                      metavar="FILE", default="train.conll")
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file",
                      metavar="FILE", default="test.conll")
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file",
                      metavar="FILE", default="test.conll")
    parser.add_option("--extrn", dest="external_embedding",
                      help="External embeddings", metavar="FILE")
    parser.add_option("--wembedding", type="int",
                      dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int",
                      dest="pembedding_dims", default=25)
    parser.add_option("--rembedding", type="int",
                      dest="rembedding_dims", default=25)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=125)
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string",
                      dest="output", default="results")
    parser.add_option("--activation", type="string",
                      dest="activation", default="tanh")
    parser.add_option("--layers", type="int",
                      dest="layers", default=2)
    parser.add_option("--dropout", type="float",
                      dest="dropout", default=0.1)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=125)
    parser.add_option("--disableblstm", action="store_false",
                      dest="blstmFlag", default=False)
    parser.add_option("--disablelabels", action="store_false",
                      dest="labelsFlag", default=True)
    parser.add_option("--predict", action="store_true",
                      dest="predictFlag", default=False)
    parser.add_option("--bibi-lstm", action="store_true",
                      dest="bibiFlag", default=False)

    (options, args) = parser.parse_args()

    print('Using external embedding:', options.external_embedding)

    if options.predictFlag:

        stored_opt.external_embedding = options.external_embedding

        print('Initializing lstm mstparser:')
        parser = model.MSTParserLSTM(words, pos, rels, w2i, stored_opt)

        parser.Load(options.model)
        conllu = (os.path.splitext(options.conll_test.lower())[1] == '.conllu')
        tespath = os.path.join(
            options.output, 'test_pred.conll' if not conllu else 'test_pred.conllu')

        ts = time.time()
        test_res = list(parser.Predict(options.conll_test, options))
        te = time.time()
        print('Finished predicting test.', te - ts, 'seconds.')
        util.write_conll(tespath, test_res)

        if not conllu:
            os.system('perl src/utils/eval.pl -g ' + options.conll_test +
                      ' -s ' + tespath + ' > ' + tespath + '.txt')
        else:
            os.system('python src/utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' +
                      options.conll_test + ' ' + tespath + ' > ' + testpath + '.txt')
    else:
        print('Preparing vocab')
        words, w2i, pos, rels = util.vocab(options.conll_train)

        print('Finished collecting vocab')

        print('Initializing lstm mstparser:')
        parser = model.MSTParserLSTM(words, pos, rels, w2i, options)

        for epoch in range(options.epochs):
            print('Starting epoch', epoch)
            Train(parser, epoch, options.conll_train)
            # conllu = (os.path.splitext(
            #     options.conll_dev.lower())[1] == '.conllu')
            # devpath = os.path.join(options.output, 'dev_epoch_' +
            #                        str(epoch + 1) + ('.conll' if not conllu else '.conllu'))
            # util.write_conll(devpath, parser.Predict(
            #     options.conll_dev, options))
            # parser.save()

            # if not conllu:
            #     os.system('perl src/utils/eval.pl -g ' + options.conll_dev +
            #               ' -s ' + devpath + ' > ' + devpath + '.txt')
            # else:
            #     os.system('python src/utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' +
            # options.conll_dev + ' ' + devpath + ' > ' + devpath + '.txt')
