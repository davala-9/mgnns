


if __name__ == '__main__':

    input = open("kinship.data", 'r')
    output = open("kinship_all.tsv", 'w') 
    for line in input.readlines():
        if line.strip(): 
            # each line is of the form relation(Name1, Name2) 
            relation = line.split('(')[0]
            subject = line.split('(')[1].split(',')[0]
            oobject = line.split(', ')[1][:-2]
            output.write("{}\t{}\t{}\n".format(subject, relation, oobject))

    input.close()
    output.close()
    

