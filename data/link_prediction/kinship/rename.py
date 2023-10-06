


if __name__ == '__main__':

    input = open("kinship_all.tsv", 'r')
    output = open("kinship_test.tsv", 'w')  
    name_change = {"Christopher": "Adam",
                   "Arthur": "Benjamin",
                   "Victoria": "Emma",
                   "Andrew": "Caleb",
                   "James": "David",
                   "Jennifer": "Olivia",
                   "Colin": "Ethan",
                   "Charlotte": "Ava",
                   "Roberto": "George",
                   "Emilio": "Henry",
                   "Lucia": "Isabella",
                   "Pierro": "Isaac",
                   "Angela": "Mia",
                   "Marco": "Jacob",
                   "Alfonso": "Kevin",
                   "Sophia": "Amelia",
                   "Penelope": "Harper",
                   "Christine": "Everlyn",
                   "Maria": "Emily",
                   "Francesca": "Elizabeth",
                   "Angela": "Avery",
                   "Gina": "Camila",
                   "Tomaso": "Liam",
                   "Margaret": "Scarlett",
                   "Charles": "Benjamin"}
    for line in input.readlines():
        s, p, o = line.split()
        output.write("{}\t{}\t{}\n".format(name_change[s], p, name_change[o]))

    input.close()
    output.close()
    

