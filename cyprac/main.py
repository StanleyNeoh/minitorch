import time
import hello
import primes as Cy

start = time.time()
Cy.primes(10000)
print(time.time() - start)

def primes(nb_primes):
    p = [0] * 10000
    len_p = 0
    n = 2
    while len_p < nb_primes:
        for i in p[:len_p]:
            if n % i == 0:
                break
        else:
            p[len_p] = n
            len_p += 1
        n += 1
    result = [x for x in p[:len_p]]
    return result

start = time.time()
primes(10000)
print(time.time() - start)