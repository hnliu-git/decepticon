from t5_inf import RaceInfModule
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--article', type=str, default='This is a test')
    parser.add_argument('--answer', type=str, default='test')
    args = parser.parse_args()

    fx_infer = RaceInfModule.load_from_checkpoint("ckpts/t5_que.ckpt")
    fx_infer.eval()

    # Single test case
    article = args.article
        # "No one knows for certain why people dream , but some dreams mi ##sh ##t be connected to the mental processes that help us learn . In a recent study , scientists found a connection between nap - time dreams and better memory in people who were learning a new skill . \" I was astonished by this finding , \" Robert Stick ##gold told Science News . He is a cognitive ne ##uro ##s ##cie ##nti ##st at Harvard Medical School who worked on the study of - how the brain and nervous system work , and cognitive studies look at how people learn and reason . So a cognitive ne ##uro ##s ##cie ##nti ##st may study the brain processes that help people learn . In the study , 99 college students between the ages of 18 and 30 each spent an hour on a computer , trying to get through a virtual maze . The maze was difficult , and the study participants had to start in a different place each time they tried - making it even more difficult . They were also told to find a particular picture of a tree and remember where it was . For the first 90 minutes of a five - hour break , half of the particular ##ity stayed awake and half were told to take a short nap . Part ##ici ##pants who stayed awake were asked to describe their thoughts . Part ##ici ##pants who took a nap were asked about their dreams before sleep and after steep - and they were awakened within a minute of sleep to describe their dreams . About a dozen of the 50 people who slept said their dreams were connected to the maze . Some dreamed about the music that had been playing when they were working ; others said they dreamed about seeing people in the maze . When these people tried the computer maze again , they were generally able to find the tree faster than before their nap ##s . However , people who had other dreams , or people who didn ' t take a nap , didn ' t show the same improvement . Stick ##gold suggests the dream itself doesn ' t help a person learn - it ' s the other way around ."
    answer = args.answer
        # 'see how dreams and learning are connected'
    questions = fx_infer.generate_sentence(article, answer)

    fx_infer = RaceInfModule.load_from_checkpoint("ckpts/t5_dis.ckpt")
    fx_infer.eval()
    print("ANS", answer)
    print("QUE", questions)
    print("DIS", fx_infer.generate_sentence(article, answer, questions))