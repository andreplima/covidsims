  # builds the roulette from the case outcome stats
  #xxx is ECO_ROULETTE_SIZE generalisable?
  (probs, recovery) = zip(*outcomes)
  thresholds = [int(sum(probs[:k+1] * ECO_ROULETTE_SIZE)) for k in range(len(probs))]
  roulette = {}
  for pocket in range(ECO_ROULETTE_SIZE + 1):
    k = 0
    while pocket > thresholds[k]: k += 1
    roulette[pocket] = recovery[k]

  def spinWheel(): # xxx add normal noise?
    pocket = randint(0, ECO_ROULETTE_SIZE)
    return roulette[pocket]


  # builds the book of life
  bol = defaultdict(lambda: defaultdict(int))
  for (territory, date, N, newCases, newDeaths) in sourceData:

    # processes the new cases
    for _ in range(newCases):

      # (1) a new active case is accounted, and ...
      bol[date][ECO_INFECTIOUS] += 1

      # (2) ... after some time, the case is resolved, for the better ...
      #dt = timedelta(days = spinWheel())
      #bol[date + dt][ECO_INFECTIOUS] -= 1
      #bol[date + dt][ECO_RECOVERED]  += 1 # P1: assumes everyone recovers

    # processes the new deaths
    for _ in range(newDeaths):

      # (3) ... or for the worse.
      #bol[date][ECO_RECOVERED] -= 1 # retracts the assumption P1
      bol[date][ECO_DECEASED]  += 1

    # xxx issues with balance

