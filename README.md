## Problem Formulation
Given problem can be formulated as a Multiple Pickup and delivery vehicle routing problem, with constraints on time window and vehicle capacity. 
* **Objective:** Objective is to minimize the total travel time. (so that all the deliveries can be done with minimum posiible time)
* **Constraints:**
 1. Pickup should happen before the delivery of the same order.
 2. Pickup and delivery of the same order should be in same vehicle.
 3. Total load in the vehicle at every node should not beyond the maximum capacity of vehicle
 4. Vehicle should reach the locations within the time window.

**Assumptions taken:** 
* Each vehicle returns to it's starting location (This is a temperory assumption which can be modified later by giving end locations for vehicles)
* All vehicle are similar in the sence of travel time.Order grouping is based only on the starting locations of vehicles.
* The orders are prior known, and the demand of each order is static.

## Solution Methodology

My solution proposes a two stage algorithm with order clustering first and then vehicle routing optimization.  Vehicle grouping is based on it's starting position. For convinience of usage, the term 'depot' is used to indicate vehicle's starting location. ( Assuming there are multiple vehicle depots for this logistics, in which vehicle's are waiting for the delivery orders). Genetic algorithm in combination of Clarke-Wright algorithm is used for vehicle routing optimization, for each groups. 

**Order Clustering**

Orders are grouped according to the pickup locations. The closest depot is calculated for each pickup locations. Then each pickup location is assigned to corresponding closest depot and orders are grouped according to which depot their pickup location belongs to. 
Closest depot can be computed according to the travel time which can be obtained from the Google Map API's distance matrix. 

**Vehicle Routing Optimization**

Clarke-Wright (CW) algorithm is used to find the initial solution. CW algorithm is a best way to findout the initial solutons for the veicle routing problems [(YongWang)](https://doi.org/10.1016/j.tre.2020.102118). Time windows and vehicle capacity constraints are considered while computing this initial soultion.Then a genetic algorithm based approach which takes this initial soultion as input is used to find the optimal solution. 

[*Detailed solution explanation*]()













