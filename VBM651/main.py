def robojump(x1, x2, v1, v2):
    while True:
        if x1 == x2:
            return True
        elif (v1 > v2 and x1 > x2) or (v2 > v1 and x2 > x1):
            return False
        x1 += v1
        x2 += v2


def contest(filename):
    with open(filename, "r") as f:
        a = f.readlines()
        b = [i.split() for i in a]
        for i in range(len(b)):
            for j in range(len(b[i])):
                b[i][j] = int(b[i][j])
        dictionary = {}
        liste = []
        for i in range(len(b)):
            dictionary[i + 1] = 0
        for j in range(len(b[0])):
            max1 = 0
            liste = []
            for i in range(len(b)):
                liste.append(b[i][j])
            max1 = max(liste)
            for i2 in range(len(liste)):
                if liste[i2] == max1:
                    dictionary[i2 + 1] += 1
        won = []
        max2 = 0
        for i3 in dictionary:
            if dictionary[i3] > max2:
                max2 = i3
        for i4 in dictionary:
            if dictionary[i4] == max2:
                won.append(i4)
        for i5 in dictionary.keys():
            print(str(i5) + ". Yarışmacı: " + str(dictionary[i5]) + " puan")
        print("Kazanan: ", end="")
        print(str(won[0]), end="")
        for i6 in range(1, len(won)):
            print(", " + str(won[i6]), end="")
        print()


def scoreboard(ranklist, scores):
    for score in scores:
        for i in range(len(ranklist)):
            j = ranklist[i]
            if score > j:
                ranklist.insert(i, score)
                counter = 0
                temp = 0
                while temp < i + 1:
                    if ranklist[temp] == score:
                        print(counter + 1)
                        break
                    if ranklist[temp] != ranklist[temp + 1]:
                        counter += 1
                    temp += 1
                break
            elif i == len(ranklist) - 1:
                print(i)
                ranklist.append(score)
