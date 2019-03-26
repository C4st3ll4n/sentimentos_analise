from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

dataset = ["Fala que odeia falsidade, mas usa Windows pirata.",
           "Como se faz para dar o logout na vida? Parece que entrei no save errado.",
           "Acho que fiz login na visda errada.",
           "No algoritmo da vida, eu sou aquele ';' que chega e estraga tudo",
           "Pessoa falsa é pior que aquele download que falha com 99%.",
           "Que vida chata... Como que formata?",
           "Abrindo o computador pra dar parabéns pra placa mãe.",
           "Seria perfeito se o Bluetooth transmitisse o amor, já que muitos olhares não fazem mais isso.",
           "Um dia tão inútil quanto a maioria dos aplicativos embutidos no Android.",
           "Aquele momento maravilhoso em que o download atinge o clímax.",
           "Antivírus é igual mãe: chato, mas só quer o seu bem.",
           "Minha internet está mais lenta que o sistema judiciário.",
           "Nada é tão ruim que a sua internet não possa piorar!",
           "É pra frente que se anda, é pra cima que se olha e é lutando que se conquista.",
           "Treine sua mente para ver o lado bom de qualquer situação.",
           "Bandido bom é bandido morto",
           "vocês vão me desculpar mas não existe estado que transmite paz melhor que a de MG",
           "o Brasil é um Belo Pais",
           "Tô feliz",
           "Só a felicidade reina",
           "Aqui maldade não se cria",
           "Que odio",
           "Amo meu trabalho"]

polaris = [-1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1]

dataTreino, dataValidacao, polTreino, polValidacao = train_test_split(dataset, polaris, test_size=0.2)
# print("dataTreino -> {} \n poltreino -> {}\n".format(dataTreino, polTreino))
bag = CountVectorizer()

bag_treino = bag.fit_transform(dataTreino)
bag_val = bag.transform(dataValidacao)

bagTreinoArray = bag_treino.toarray()
bagValidaArray = bag_val.toarray()

nBayes = MultinomialNB()
nBayes.fit(bagTreinoArray, polTreino)

predTreino = nBayes.predict(bagTreinoArray)
predVal = nBayes.predict(bagValidaArray)

# print("Polariades preditas ( treinamento ) :\n{}".format(predTreino))
print("Acurácia no treinamento: {}".format(accuracy_score(polTreino, predTreino)))

# print("Polariades preditas ( validacao ) :\n{}".format(predVal))
print("Acurácia na validação: {}".format(accuracy_score(polValidacao, predVal)))

while True:
    frase_teste = [str(input("Insira uma frase: "))]
    bag_teste = bag.transform(frase_teste)
    predicaoPolTeste = nBayes.predict(bag_teste.toarray())
    print("Sentimento positivo" if predicaoPolTeste == 1 else "Sentimento negativo")
    print(predicaoPolTeste)
    op = str(input("Deseja continuar? [ S / N ]\nR:"))
    if op[0].upper() == "S":
        novoDataset = dataset + frase_teste
        novoPolaris = polaris + predicaoPolTeste
        bagNovoTreino = bag.transform(novoDataset)
        novaPredicao = nBayes.predict(bagNovoTreino.toarray())
        # print("Novo bag: {}\nNova Predição: {}\n "bagNovoTreino, novaPredicao)
    else:
        break
