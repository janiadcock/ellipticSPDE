import "regent"

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c

-------------------------------------------------------------------------------
-- TO DO: change these to be inputs from a json file so can run multiple samples at once w/ diff. inputs
-------------------------------------------------------------------------------
local NUM_GRID_POINTS = 7
local k = 0.17
--local k = C.drand48()

-------------------------------------------------------------------------------
-- CONSTANTS
-------------------------------------------------------------------------------
local PI = 3.1415926535898
local DOMAIN_LENGTH = 1.0 -- length of the domain (starts at 0)
local U_0 = 0.0 -- set left boundary value for unknowns
local U_1 = 0.0 -- set right boundary value for unknowns
local F = -1.0 -- set forcing term value

-------------------------------------------------------------------------------
-- TASKS
-------------------------------------------------------------------------------
-- TDMA: a, b, c, d, c_star, and d_star include boundary points
local fspace Node {
  x : double;
  k_coeff : double;
  a : double; 
  b : double;
  c : double;
  d : double;
  c_star : double;
  d_star : double; 
  u : double;
}

__demand(__leaf, __parallel,  __cuda)
task set_xk(nodes: region(ispace(int1d),Node))
where
  writes(nodes.k_coeff),
  reads writes(nodes.x)
do
  var grid_spacing = DOMAIN_LENGTH/(NUM_GRID_POINTS-1.0);
  --C.printf('grid_spacing %f\n', grid_spacing)
  __demand(__openmp)
  for node in nodes do
    var j = int1d(node)
    var i = int(j)
    --C.printf('i %d \n', i)
    --C.printf('i %f \n', i*grid_spacing)
    node.x = i*grid_spacing
    node.k_coeff = 1.0 + k*node.x
  end
end

__demand(__leaf, __cuda)
task set_abc(nodes: region(ispace(int1d),Node))
where
  reads(nodes.{x, k_coeff}),
  writes(nodes.{a, b, c})
do
  __demand(__openmp)
  for node in nodes do
    var i = int1d(node)
    node.a = (0.5*(node.k_coeff + nodes[i-1].k_coeff) / (node.x-nodes[i-1].x)) / (0.5*(node.x - nodes[i-1].x));
    node.b = ((-1.0)*0.5*(nodes[i+1].k_coeff + node.k_coeff) / (nodes[i+1].x - node.x) + (-1.0)*0.5*(node.k_coeff + nodes[i-1].k_coeff) / (node.x - nodes[i-1].x)) / (0.5*(nodes[i+1].x - nodes[i-1].x));
    node.c = (0.5*(nodes[i+1].k_coeff + node.k_coeff) / (nodes[i+1].x - node.x)) / (0.5*(nodes[i+1].x - nodes[i-1].x)); 
  end
end

__demand(__leaf)
task set_cd_star(nodes: region(ispace(int1d),Node))
where
  reads(nodes.{a, b, c, d}),
  reads writes(nodes.{c_star, d_star})
do
  var m = 0.0;
  for node in nodes do
    var i = int1d(node)
    if i == nodes.ispace.bounds.lo then
      node.c_star = node.c / node.b;
      node.d_star = node.d / node.b;
    elseif i == nodes.ispace.bounds.hi then
      m = 1.0/(node.b - node.a * nodes[i-1].c_star);
      node.d_star = (node.d - node.a * nodes[i-1].d_star)*m; 
    else
      m = 1.0/(node.b - node.a * nodes[i-1].c_star);
      node.c_star = node.c * m;
      node.d_star = (node.d - node.a * nodes[i-1].d_star)*m;
    end
  end
end

__demand(__leaf, __cuda)
task calc_u(nodes: region(ispace(int1d),Node))
where
  reads(nodes.{c_star, d_star}),
  reads writes(nodes.u)
do
  __demand(__openmp)
  for node in nodes do
    var i = int1d(node)
    if i == nodes.ispace.bounds.hi then
      node.u = node.d_star
    else
      node.u = node.d_star - node.c_star * nodes[i+1].u
    end 
  end
end

-- change to reduces
__demand(__leaf, __parallel, __cuda)
task integrate_u(nodes: region(ispace(int1d), Node)) : double
where
  reads(nodes.u)
do
  var sum = 0.0;
  for node in nodes do
    sum += node.u
  end
  return sum
end

-------------------------------------------------------------------------------
-- MAIN

-- Solve the 1D diffusion eqn. with an uncertain variable coefficient
-- using finite differences and TDMA.
-- del(k del(u))=f on [0,1] subject to u(0)=0 and u(1)=0
-- Here we set f=-1 and k is a random diffusivity
-------------------------------------------------------------------------------
task main()
  var index_space = ispace(int1d, {NUM_GRID_POINTS})
  var nodes = region(index_space, Node) 
  C.printf("after var nodes\n")

  fill(nodes.d, F)
  C.printf("after fill\n")

  set_xk(nodes)
  C.printf("after set_xk\n")

  --var c0 = ispace(int1d, NUM_GPUS)
  --var p0 = partition(equal, nodes, c0)
  
  set_abc(nodes)

  -- correct boundary nodes 
  nodes[0].a = 0.0;
  nodes[0].b = 1.0;
  nodes[0].c = 0.0;
  nodes[0].d = U_0;
  nodes[NUM_GRID_POINTS-1].a = 0.0;
  nodes[NUM_GRID_POINTS-1].b = 1.0;
  nodes[NUM_GRID_POINTS-1].c = 0.0;
  nodes[NUM_GRID_POINTS-1].d = U_1;

  C.printf("after set_abc\n")
 
  __fence(__execution, __block)
  C.printf('x, a, b, c, d\n')
  for node in nodes do
    C.printf('%f %f %f %f %f \n', node.x, node.a, node.b, node.c, node.d)
  end
  __fence(__execution, __block)

  set_cd_star(nodes)
  C.printf("after set_cd_star\n")

  calc_u(nodes)
  C.printf("after calc_u\n")

  var sum = integrate_u(nodes)

  __fence(__execution, __block)
  C.printf('sum = %f \n', sum)
end

regentlib.saveobj(main, "diffusion.o")
