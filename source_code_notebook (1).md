```python
import random
import math
import numpy as np

# Initial declarations of some variables
population_size = 25 

depots = None
nodes = None
population = None
```

Two classes are defined, each for Depot and Node details. (here I am using the term node for pickup or delovery locations).

To simplify the solution process, I have assumed the time windows associated with an order as the service time of the corresponding delivery or pickup locations. Also it is assumed that all the vehicle at a depot have same maximum load capacity


```python
class Depot:

    def __init__(self, max_vehicles,max_load):
        self.pos = (0, 0)
        self.max_vehicles = max_vehicles 
        self.max_load = max_load #maximum load allowed for a single vehicle in a depot
        self.closest_nodes = []


class Node:

    def __init__(self, id, x, y, demand):
        self.id = id #name or identification id given for the nodes
        self.pos = (x, y)
        self.service_duration = service_duration
        self.demand = demand

```

**Distance Matrix**

To calculate the travel time, Google map distance amtrix API can be used. Here I have assumed that the travel time is proportional to the distance and distace is calculated using the formula. This is not the practical case. But this function can be modified later easily, and remaining code will not be changed. However, along the code, I have used 'distance' instead of 'travel time'.

input to the distance matrix is the position of two locations as x and y coordinates. (latitude,longitude)


```python
def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
```

A function to set population size


```python
def set_population_size(size):
    population_size = size
```

**Finding the Closest depots**


```python
def find_closest_depot(pos):
    closest_depot = None
    closest_distance = -1
    for i, depot in enumerate(depots):
        d = distance(depot.pos, pos)
        if closest_depot is None or d < closest_distance:
            closest_depot = (depot, i)
            closest_distance = d

    return closest_depot[0], closest_depot[1], closest_distance # depotposition,depotid,distance
```

**Grouping**


```python
# Group pickup locations to closest depot
for n in Nodes:
    if Nodes.type == 'PL': #pickup location 
        depot, depot_index, dist = find_closest_depot(n.pos)
        groups[depot_index].append(n.id)
```


```python
#Group delivery locations to corresponding pickup locations
for n in Nodes:
    if Nodes.type == 'DL'
    for i in (orders):
        if i.dl == n:
            depot, depot_index, dist = find_closest_depot(i.pl_pos)
            groups[depot_index].append(n.id)
```

**Initial Solution**


```python
# a random chromosome
def create_random_chromosome(groups):
    routes = []
    for d in range(len(groups)):
        depot = depots[d]
        group = groups[d][:]
        random.shuffle(group)
        routes.append([[]])

        r = 0
        route_cost = 0
        route_load = 0
        last_pos = depot.pos
        for c in group:
            customer = customers[c - 1]
            cost = distance(last_pos, node.pos) + node.service_duration + find_closest_depot(node.pos)[2]
            if  route_load + node.demand > depot.max_load:
                r += 1
                routes[d].append([])
            routes[d][r].append(c)
```


```python
#greedy heuristic algorithm using CW savings method
def create_heuristic_chromosome(groups):
    routes = [[] for i in range(len(depots))]
    missing_nodes = list(map(lambda x: x.id,nodes))
    for d in range(len(groups)):
        depot = depots[d]
        savings = []
        for i in range(len(groups[d])):
            ni = nodess[groups[d][i] - 1]
            savings.append([])
            for j in range(len(groups[d])):
                if j <= i:
                    savings[i].append(0)
                else:
                    nj = nodes[groups[d][j] - 1]
                    #calculate savings in the sence of travel distance
                    savings[i].append(distance(depot.pos, ni.pos) + distance(depot.pos, nj.pos) -
                                      distance(ni.pos, nj.pos)) 
        savings = np.array(savings)
        order = np.flip(np.argsort(savings, axis=None), 0) # sorting in descending order

        for saving in order:
            i = saving // len(groups[d])
            j = saving % len(groups[d])

            ni = groups[d][i]
            nj = groups[d][j]

            ri = -1
            rj = -1
            for r, route in enumerate(routes[d]):
                if ni in route:
                    ri = r
                if nj in route:
                    rj = r

            route = None
            if ri == -1 and rj == -1:
                if len(routes[d]) < depot.max_vehicles:
                    route = [ni, nj]
            elif ri != -1 and rj == -1:
                if routes[d][ri].index(ni) in (0, len(routes[d][ri]) - 1):
                    route = routes[d][ri] + [nj]
            elif ri == -1 and rj != -1:
                if routes[d][rj].index(nj) in (0, len(routes[d][rj]) - 1):
                    route = routes[d][rj] + [ni]
            elif ri != rj:
                route = routes[d][ri] + routes[d][rj]

            if route:
                if is_consistent_route(route, depot, True)[1] == 2: # checking the consitancy of route
                    route = schedule_route(route)
                if is_consistent_route(route, depot):
                    if ri == -1 and rj == -1:
                        routes[d].append(route)
                        missing_nodes.remove(ni)
                        if ni != nj:
                            missing_nodes.remove(nj)
                    elif ri != -1 and rj == -1:
                        routes[d][ri] = route
                        missing_nodes.remove(nj)
                    elif ri == -1 and rj != -1:
                        routes[d][rj] = route
                        missing_nodes.remove(ni)
                    elif ri != -1 and rj != -1:
                        if ri > rj:
                            routes[d].pop(ri)
                            routes[d].pop(rj)
                        else:
                            routes[d].pop(rj)
                            routes[d].pop(ri)
                        routes[d].append(route)


    # Order nodes within routes
    for i, depot_routes in enumerate(routes):
        for j, route in enumerate(depot_routes):
            new_route = schedule_route(route) #calling the route scheduling function
            routes[i][j] = new_route

    chromosome = encode(routes)
    chromosome.extend(missing_nodes)
    return chromosome

```


```python
def initialize(random_portion=0):
    global population
    population = []
    groups = [[] for i in range(len(depots))]

    # Group customers to closest depot
    for c in customers:
        depot, depot_index, dist = find_closest_depot(c.pos)
        groups[depot_index].append(c.id)

    for z in range(int(population_size * (1 - random_portion))):
        chromosome = create_heuristic_chromosome(groups)
        population.append((chromosome, evaluate(chromosome)))

    for z in range(int(population_size * random_portion)):
        chromosome = create_random_chromosome(groups)
        population.append((chromosome, evaluate(chromosome)))
```

**Route consitency**


```python
def is_consistent_route(route, depot, include_reason=False):
    route_load = 0
    route_duration = 0
    last_pos = depot.pos
    for c in route:
        node = nodes[c - 1]
        route_load += node.demand
        route_duration += distance(last_pos, node.pos) + node.service_duration
        last_pos = node.pos
    route_duration += find_closest_depot(last_pos)[2]

    if include_reason:
        if route_load > depot.max_load: 
            return False, 1
        return True, 0
    return route_load <= depot.max_load and (depot.max_duration == 0 or route_duration <= depot.max_duration)
```


```python
# a chromosome (the final route) is consistent only when the number of groups generated (vehicles used) is less than the maximum number of vehcicles in a depot
def is_consistent(chromosome):
    for c in node:
        if c.id not in chromosome:
            return False

    routes = decode(chromosome)
    for d in range(len(routes)):
        depot = depots[d]
        if len(routes[d]) > depot.max_vehicles:
            return False
        for route in routes[d]:
            if not is_consistent_route(route, depot):
                return False

    return True
```

**chromosome encoding and decoding**


```python
def encode(routes):
    chromosome = []
    for d in range(len(routes)):
        if d != 0:
            chromosome.append(-1)
        for r in range(len(routes[d])):
            if r != 0:
                chromosome.append(0)
            chromosome.extend(routes[d][r])
    return chromosome


def decode(chromosome):
    routes = [[[]]]
    d = 0
    r = 0
    for i in chromosome:
        if i < 0:
            routes.append([[]])
            d += 1
            r = 0
        elif i == 0:
            routes[d].append([])
            r += 1
        else:
            routes[d][r].append(i)
    return routes

```

**Evaluation of chromosome**


```python
def evaluate(chromosome, return_distance=False):
    for c in nodes:
        if c.id not in chromosome:
            if return_distance:
                return math.inf
            return 0

    routes = decode(chromosome)
    score = 0
    for depot_index in range(len(routes)):
        depot = depots[depot_index]
        for route in routes[depot_index]:
            route_length, route_load = evaluate_route(route, depot, True)

            score += route_length

            ##Sometimes penalty for the violation of tw and capacity constraints should be added here. Now I am not sure about it.
           
    if return_distance:
        return score
    return 1/score
```


```python
#Evaluation of route 
def evaluate_route(route, depot, return_load=False):
    if len(route) == 0:
        if return_load:
            return 0, 0
        return 0
    route_load = 0
    route_length = 0
    node = None
    last_pos = depot.pos
    for cid in route:
        node = nodes[cid - 1]
        route_load += node.demand
        route_length += distance(last_pos, node.pos)
        route_length += node.service_duration
        last_pos = node.pos
    route_length += find_closest_depot(node.pos)[1]

    if return_load:
        return route_length, route_load
    return route_length
```

**Scheduling the route**


```python
def schedule_route(route):
    if not len(route):
        return route
    new_route = []
    prev_node = random.choice(route)
    route.remove(prev_cust)
    new_route.append(prev_node)

    while len(route):
        prev_node = min(route, key=lambda x: distance(node[x - 1].pos, node[prev_cust - 1].pos))
        route.remove(prev_node)
        new_route.append(prev_node)
    return new_route
```

**Selection of route, (elite selection)**


```python
def select(portion, elitism=0):
    total_fitness = sum(map(lambda x: x[1], population))
    weights = list(map(lambda x: (total_fitness - x[1])/(total_fitness * (population_size - 1)), population))
    selection = random.choices(population, weights=weights, k=int(population_size*portion - elitism))
    population.sort(key=lambda x: -x[1])
    if elitism > 0:
        selection.extend(population[:elitism])
    return selection
```

**CrossOver**


```python
def crossover(p1, p2):
    protochild = [None] * max(len(p1), len(p2))
    cut1 = int(random.random() * len(p1))
    cut2 = int(cut1 + random.random() * (len(p1) - cut1))
    substring = p1[cut1:cut2]

    for i in range(cut1, cut2):
        protochild[i] = p1[i]

    p2_ = list(reversed(p2))
    for g in substring:
        if g in p2_:
            p2_.remove(g)
    p2_.reverse()

    j = 0
    for i in range(len(protochild)):
        if protochild[i] is None:
            if j >= len(p2_):
                break
            protochild[i] = p2_[j]
            j += 1

    i = len(protochild) - 1
    while protochild[i] is None:
        protochild.pop()
        i -= 1

    population.append((protochild, evaluate(protochild)))
```

**Mutations**


```python
def heuristic_mutate(p):
    g = []
    for i in range(3):
        g.append(int(random.random() * len(p)))

    offspring = []
    for i in range(len(g)):
        for j in range(len(g)):
            if g == j:
                continue
            o = p[:]
            o[g[i]], o[g[j]] = o[g[j]], o[g[i]]
            offspring.append((o, evaluate(o)))

    selected_offspring = max(offspring, key=lambda o: o[1])
    population.append(selected_offspring)







```


```python
def inversion_mutate(p):
    cut1 = int(random.random() * len(p))
    cut2 = int(cut1 + random.random() * (len(p) - cut1))

    if cut1 == cut2:
        return
    if cut1 == 0:
        child = p[:cut1] + p[cut2 - 1::-1] + p[cut2:]
    else:
        child = p[:cut1] + p[cut2 - 1:cut1 - 1:-1] + p[cut2:]
    population.append((child, evaluate(child)))

```


```python
def best_insertion_mutate(p):
    g = int(random.random() * len(p))

    best_child = None
    best_score = 0

    for i in range(len(p) - 1):
        child = p[:]
        gene = child.pop(g)
        child.insert(i, gene)
        score = evaluate(child)
        if score > best_score:
            best_score = score
            best_child = child

    population.append((best_child, best_score))
```


```python

def depot_move_mutate(p):
    if -1 not in p:
        return
    i = int(random.random() * len(p))
    while p[i] != -1:
        i = (i + 1) % len(p)

    move_len = int(random.random() * 10) - 5
    new_pos = (i + move_len) % len(p)

    child = p[:]
    child.pop(i)
    child.insert(new_pos, -1)
    population.append((child, evaluate(child)))

```

**Merging**


```python
def route_merge(p):
    routes = decode(p)

    d1 = int(random.random() * len(routes))
    r1 = int(random.random() * len(routes[d1]))
    d2 = int(random.random() * len(routes))
    r2 = int(random.random() * len(routes[d2]))

    if random.random() < 0.5:
        limit = int(random.random() * len(routes[d2][r2]))
    else:
        limit = len(routes[d2][r2])

    reverse = random.random() < 0.5

    for i in range(limit):
        if reverse:
            routes[d1][r1].append(routes[d2][r2].pop(0))
        else:
            routes[d1][r1].append(routes[d2][r2].pop())
    routes[d1][r1] = schedule_route(routes[d1][r1])
    routes[d2][r2] = schedule_route(routes[d2][r2])
    child = encode(routes)
    population.append((child, evaluate(child)))
```

**Combining all functions to run algorithm**


```python
def train(generations, crossover_rate, heuristic_mutate_rate, inversion_mutate_rate,
          depot_move_mutate_rate, best_insertion_mutate_rate, route_merge_rate,
          intermediate_plots=False, log=True):
    global population
    for g in range(generations):
        if log and g % 10 == 0:
            best = max(population, key=lambda x: x[1])
            print(f'[Generation {g}] Best score: {best[1]} Consistent: {is_consistent(best[0])}')

        selection = select(heuristic_mutate_rate + inversion_mutate_rate
                           + crossover_rate + depot_move_mutate_rate + best_insertion_mutate_rate
                           + route_merge_rate)
        selection = list(map(lambda x: x[0], selection))

        offset = 0
        for i in range(int((population_size * crossover_rate) / 2)):
            p1, p2 = selection[2*i + offset], selection[2*i + 1 + offset]
            crossover(p1, p2)
            crossover(p2, p1)
        offset += int(population_size * crossover_rate)

        for i in range(int(population_size * heuristic_mutate_rate)):
            heuristic_mutate(selection[i + offset])
        offset += int(population_size * heuristic_mutate_rate)

        for i in range(int(population_size * inversion_mutate_rate)):
            inversion_mutate(selection[i + offset])
        offset += int(population_size * inversion_mutate_rate)

        for i in range(int(population_size * depot_move_mutate_rate)):
            depot_move_mutate(selection[i + offset])
        offset += int(population_size * depot_move_mutate_rate)

        for i in range(int(population_size * best_insertion_mutate_rate)):
            best_insertion_mutate(selection[i + offset])
        offset += int(population_size * best_insertion_mutate_rate)

        for i in range(int(population_size * route_merge_rate)):
            route_merge(selection[i + offset])
        offset += int(population_size * route_merge_rate)

        population = select(1.0, elitism=4)


    best_solution = None
    if is_consistent(population[0][0]):
        print(f'Best score: {population[0][1]}, best distance: {evaluate(population[0][0], True)}')
        best_solution = population[0][0]
    else:
        for c in population:
            if is_consistent(c[0]):
                print(f'Best score: {c[1]}, best distance: {evaluate(c[0], True)}')
                best_solution = c[0]
                break
        else:
            print('No consistent solution found!!!!.')
    if best_solution:
        plot(best_solution)
    return best_solution
```

The algorithm can be run by calling required functions as follows.


```python
generations = #2500
population_size =# 50
crossover_rate = #0.05
heuristic_mutate_rate =# 0.05
inversion_mutate_rate = #0.05
depot_move_mutate_rate = #0
best_insertion_mutate_rate =# 0.1
route_merge_rate =# 0.05

set_population_size(population_size)
initialize()
best_solution =train(generations, crossover_rate, heuristic_mutate_rate, inversion_mutate_rate,
                                  depot_move_mutate_rate, best_insertion_mutate_rate, route_merge_rate,
                                  intermediate_plots=True)

```
