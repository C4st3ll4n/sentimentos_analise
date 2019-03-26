from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

dataset = ["Eu amo este carro.",
           "Este visual é lindo.",
           "Eu gosto muito da noite.",
           "Estou muito ansioso para o show.",
           "Ela é minha grande amiga.",
           "Eu não gosto deste carro.",
           "Este visual é horrível.",
           "Eu me sinto cansada nesta tarde.",
           "Não estou ansioso para a viagem.",
           "Ele é muito organizado.",
           "Eu me sinto feliz hoje.",
           "Ela é muito inteligente e muito dedicada.",
           "Eu não confio naquele homem.",
           "Minha casa está suja.",
           "Seu vizinho é chato."]

polaris = [1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1]

dataTreino, dataValidacao, polTreino, polValidacao = train_test_split(dataset, polaris, test_size=0.30)

# print("dataTreino -> {}\ndataVal -> {}\npolTreino->{}\npolVal->{}"
# .format(dataTreino, dataValidacao, polTreino, polTreino))

bag = CountVectorizer()

bagTreino = bag.fit_transform(dataTreino)
bagValida = bag.transform(dataValidacao)

# print("Vocabulário aprendido: \n {}\n".format(sorted(bag.vocabulary_)))
# print(bagTreino)

bagTreinoArray = bagTreino.toarray()
bagValArray = bagValida.toarray()

naiveBayes = MultinomialNB()
naiveBayes.fit(bagTreinoArray, polTreino)

predicaoPolsTreino = naiveBayes.predict(bagTreinoArray)
predicaoPolsVal = naiveBayes.predict(bagValArray)

# print("Polariades preditas ( treinamento ) :\n{}".format(predicaoPolsTreino))
# print("Acurácia no treinamento: {}".format(accuracy_score(polTreino, predicaoPolsTreino)))

# print("Polariades preditas ( validacao ) :\n{}".format(predicaoPolsVal))
# print("Acurácia na validação: {}".format(accuracy_score(polValidacao, predicaoPolsVal)))

while True:
    frase_teste = [str(input("Insira uma frase: "))]
    bag_teste = bag.transform(frase_teste)
    predicaoPolTeste = naiveBayes.predict(bag_teste.toarray())
    print("Sentimento positivo" if predicaoPolTeste == 1 else "Sentimento negativo")
    op = str(input("Deseja continuar? [ S / N ]\nR:"))
    if op[0].upper() == "S":
        pass
    else:
        break
