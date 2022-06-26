import gensim.downloader as api

model = api.load("conceptnet-numberbatch-17-06-300")

while True:
    s = input("Enter word:")
    try:
        print(s, model.most_similar(s))
    except Exception as e:
        print("Error", e)
