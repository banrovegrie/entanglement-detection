### Study
- [ ]  Literature survey
    - [x]  2xd → Confirm the solution
    - [x]  3x3 → Has it been solved? Not solved
    - [ ]  Maps for constructing absolutely separable states
    - [ ]  Find if the ABSEP channel is biased → one closed set or not?
- [ ]  Entanglement review
    - [ ]  Entanglement measure → Why not solved entanglement detection?
    - [ ]  QI for entropy stuff
    - [x]  Witness
    - [ ]  Why Tr(W rho) >= 0 and < 0 across the hyperplane set by W
    - [x]  SEP problem is NP-hard
- [ ]  Machine Learning review
    - [ ]  Supervised ML models
    - [ ]  Unsupervised ML models 
    - [x]  Compare CV algorithms for the density matrix
    - [ ]  Study convex optimisation
    - [x]  Variational circuits for state generation

### Tasks
- [x]  Write naive model
- [x]  Unitary classifier  
- [x]  Data for separable vs entangled
- [ ]  Verify entanglement classifier
- [ ]  Generate entangled matrices
- [ ]  Data for absolutely separable state classifier
- [x]  Mapping density matrix → vector (embedding)
- [ ]  Write better models (if required)
- [ ]  Put mix(ture) in different models
- [ ]  Use extra criteria for creating features (like entanglement measures)

### Meeting notes
- [ ]  Generation of data for absolutely separable states (validation)
    - Use random unitaries to generate data with high confidence
    - Compare deviations wrt the known $2 \times 2$ case
    - Try for $3 \times 3$ case (figure out if $\exists$ entangled Absolutely PPT states)
- [x]  Implementing pseudo-siamese NN for unsupervised learning
- [ ]  Figure out how to get a description of the classification boundary from the classifier
- [ ]  Further work on Implementation of a classifier for SEP based on extendibility (based on semidefinite programming definition of the problem)
- [ ]  ACVENN
- [ ]  Model based on absolute unsteerability
