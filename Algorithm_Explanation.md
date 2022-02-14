## Problem Loading
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

## Grouping or Clustering
For each pickup node, travel distance(travel time) to the depots are calculated.  Then the depot with less travel time is assigned as  closest depot. 
For each delivery location, system will check for the corresponding pickup locations and assign the DL into the group in whick the correspinding PL belongs to. 

## Vehicle Routing Optimization
Initial solution is computed using CW Algorithm.

**Steps for CW Algorithm:**

1. Assign initial distance saving values and routes (as many as number of nodes)
2. Assign vehicles to each location
3. Calculate distance savings of new vehicle route formed by any two routes (savings = d_io + d_oj - d_ij ; i,j >= 1 and i != j , here o represent depot, i,j index of nodes)
4. Sort these distance savings in descending order
5. Check for constraints (capacity and time window), if new route meets the constraints, go to step 6, o/w n=n+1 and repeat step 5
6. Generate new vehicle route, whise distance saving are maximum
7. Update distance savings through fusion of vehicles
8. Generate new route with two routes, whose savings are maximum.
9. n=1
10. Check for constraints,if met, go to step 11. o/w return to step 7
11. n=n+1
12. Generate  new vehicle route, whise distance saving are maximum
13. If there exist a route that serves only one, then go back to step 7, o/w step 4
14. Output service route of each vehicle

**Genetic Algorithm based approach to optimize route**

Solution obtained form the CW algorithm is used as the initial solution. That is to initialize the pospulation, CW algorithm is used. 

Please visit this [Notebook]() for source code explanation














