def primes(int nb_primes):
    cdef int n, i, len_p
    cdef int[10000] p
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