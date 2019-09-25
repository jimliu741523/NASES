# Neural Architecture Search in Embedding Space

This is the Code for the Paper Neural Architecture Search in Embedding Space.

### To train architectures simulator and decoder:

  sh scripts/NASES_simulating.sh
  
### To discover the CNN architectures for CIFAR-10:

  sh scripts/NASES_search.sh
  
### To run the CNN architectures by final architecture:

  sh scripts/NASES_final.sh
  
Can modify hyperparameter of 'origin_len', 'embedding' and 'addFilter' settings through main.py.

* origin_len: length of origin architecture vector.
* embedding: length of architecture-embedding vector.
* addFilter: add filter on final architecture.

The paper final architecture setting were: origin_len=60, embedding=20, addFilter=150. (The hyperparameter of addFilter please modify after process of architectures searching).
