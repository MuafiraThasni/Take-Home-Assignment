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

My solution proposes a two stage algorithm with order clustering first and then vehicle routing optimization.  Vehicle grouping is based on it's starting position. For convinience of usage, the term 'depot' is used to indicate vehicle's starting location. ( Assuming there are multiple vehicle depots for this logistics, in which vehicle's are waiting for the delivery orders). Genetic algorithm in combination of Clarke-Wright algorithm is used for vehicle routing optimization, for each groups[(MDPDPRS)](https://doi.org/10.1155/2021/5182989)

### Loading the Problem Inputs
It is considered that the system will accept the user inputs, consisiting of Order Information and Driver or Vehicle Information.

So input will contain :

**i. Order Information**
1. Orderid
2. Pickup Time Window (PT_l, PT_u)
3. Delivery Time Window (DT_l,DT_u)
4. Pickup Location Id
5. Pickup Location Position(x,y)
6. Delivery Location Id
7. Delivery Location (x,y)
8. Order Demand

**ii. Driver Information**
1. Maximum load (allowed for the vehicle)
3. Driver Start Location (x,y) (depot)
4. Driver End Location (Optional) (For the sake of simplicity, as it is now, I have considered as vehicle returns to it's starting location, and there is no need to input
vehicle end location. But this can be added into the algorithm later on improvement. )

From the input data, we can formulate the problem as a graph of (N,E),where nodes are the pickup or delivery locations and edges are the path connecting these locations. Each node 
has an assosiated data set which contains position, type (pickup/delivery), closest depot, closest distance, order details inclusing demand and time window for each order. 
First, the nodes should be clustered accorsing to the closest depot and the type. Then the vehicle routing optimization is done. 

Type of the node is decided by the demand amount. If the node is a pickup location for an arbitrary order 'o', then the corresponding demand 'demand_o' is equals to Order Demand.
For the case of delivery location, it is defined as -(Order Demand). This will allow the system to decide the type of node according to the demand sign. 

### Grouping or Clustering of Orders

Orders are grouped according to the pickup locations. The closest depot is calculated for each pickup locations. Then each pickup location is assigned to corresponding closest depot and orders are grouped according to which depot their pickup location belongs to. 
Closest depot can be computed according to the travel time which can be obtained from the Google Map API's distance matrix. 

For each pickup node, travel distance(travel time) to the depots are calculated.  Then the depot with less travel time is assigned as  closest depot. 
For each delivery location, system will check for the corresponding pickup locations and assign the DL into the group in whick the correspinding PL belongs to. 

### Vehicle Routing Optimization

Clarke-Wright (CW) algorithm is used to find the initial solution. CW algorithm is a best way to findout the initial solutions for the veicle routing problems [(GLLRP)](https://doi.org/10.1016/j.tre.2020.102118). Time windows and vehicle capacity constraints are considered while computing this initial soultion.Then a genetic algorithm based approach which takes this initial soultion as input is used to find the optimal solution. 


Please visit this [Notebook for the algorithm](https://github.com/MuafiraThasni/Take-Home-Assignment/blob/main/source_code_notebook.md). 














