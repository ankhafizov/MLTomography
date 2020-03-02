import shelve

def generate_key(shape, porsty, blobns, noise_prob, angles, tag):
    return f"dim{len(shape)},porsty{porsty},blobns{blobns},noise{noise_prob},angles{angles}_{tag}"


def show_keys():
    db = shelve.open('database')
    print(list(db.keys()))
    db.close()


def delete_element(key):
    db = shelve.open('database')
    del db[key]
    db.close()


def open(key):
    db = shelve.open('database')
    files = db[key]
    db.close()
    return files