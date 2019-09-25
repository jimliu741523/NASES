def train(embedding,origin_len):
    from DNNcoder import DNNcoder 
    import numpy as np
    import random
    import heapq
    
    DNNcoder = DNNcoder(embedding=embedding,origin_len=origin_len)
    
    
    #simulation data
    simulation_sequence = []
    num_data = 300000

    for i in range(num_data):
        foo = [i for i in range(1, 30)]*4
        simulation_sequence.append(random.sample(foo,origin_len))

    simulation_sequence = np.reshape(simulation_sequence,[num_data,origin_len])

    #test simulation data


    test_simulation_sequence = []
    num_data = 10000

    for i in range(num_data):
        foo = [i for i in range(1, 30)]*4
        test_simulation_sequence.append(random.sample(foo,origin_len))

    test_simulation_sequence = np.reshape(test_simulation_sequence,[num_data,origin_len])

    c = list(range(0, num_data))

    total_ts = [1000]
    size = 64


    for i in range(200000):
        simple = 0
        shuffle_indices = np.random.permutation(np.arange(num_data))
        simulation_sequence = simulation_sequence[shuffle_indices]
        code = DNNcoder.code(simulation_sequence[simple:size+simple])
        DNNcoder.train(simulation_sequence[simple:size+simple],simulation_sequence[simple:size+simple],code,1)
        if i%1000 == 0:
            ts = DNNcoder.loss(test_simulation_sequence[:],test_simulation_sequence[:],code,1)

            print("size:",size,i,
                   "train:",DNNcoder.loss(simulation_sequence[simple:size+simple],simulation_sequence[simple:size+simple],code,1),
                   "test:",ts)
            total_ts.append(ts)
            if heapq.nsmallest(2, total_ts)[1] > ts:
                DNNcoder.save()

    size = 128
    for i in range(100000):
        simple = 0
        shuffle_indices = np.random.permutation(np.arange(num_data))
        simulation_sequence = simulation_sequence[shuffle_indices]
        code = DNNcoder.code(simulation_sequence[simple:size+simple])
        DNNcoder.train(simulation_sequence[simple:size+simple],simulation_sequence[simple:size+simple],code,1)

        if i%1000 == 0:
            ts = DNNcoder.loss(test_simulation_sequence[:],test_simulation_sequence[:],code,1)

            print("size:",size,i,
                  "train:",DNNcoder.loss(simulation_sequence[simple:size+simple],simulation_sequence[simple:size+simple],code,1),
                  "test:",ts)
            total_ts.append(ts) 

            if heapq.nsmallest(2, total_ts)[1] > ts:
                DNNcoder.save()


    size = 256
    for i in range(50000):
        simple = 0
        shuffle_indices = np.random.permutation(np.arange(num_data))
        simulation_sequence = simulation_sequence[shuffle_indices]
        code = DNNcoder.code(simulation_sequence[simple:size+simple])
        DNNcoder.train(simulation_sequence[simple:size+simple],simulation_sequence[simple:size+simple],code,1)
        if i%1000 == 0:
            ts = DNNcoder.loss(test_simulation_sequence[:],test_simulation_sequence[:],code,1)

            print("size:",size,i,
                  "train:",DNNcoder.loss(simulation_sequence[simple:size+simple],simulation_sequence[simple:size+simple],code,1),
                  "test:",ts)
            total_ts.append(ts)
            if heapq.nsmallest(2, total_ts)[1] > ts:
                DNNcoder.save()