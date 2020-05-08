import "regent"

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c

-------------------------------------------------------------------------------
-- TO DO: change these to be inputs from a json file so can run multiple samples at once w/ diff. inputs
-------------------------------------------------------------------------------
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
local MAX_NUM_GPUS = 20

-------------------------------------------------------------------------------
-- TASKS
-------------------------------------------------------------------------------
-- TDMA: a, b, c, d, c_star, and d_star include boundary points
local fspace Node {
  x : double;
  k_coeff : double;
  l_x : double;
  r_x : double;
  l_k_coeff : double;
  r_k_coeff : double;
  a : double; 
  b : double;
  c : double;
  d : double;
  c_star : double;
  d_star : double; 
  u : double;
  lb : int;
  ub : int;
}

__demand(__leaf, __parallel,  __cuda)
task set_xk(nodes: region(ispace(int1d),Node), grid_spacing: double)
where
  writes(nodes.k_coeff),
  reads writes(nodes.x)
do
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
task set_abc(nodes: region(ispace(int1d),Node), num_grid_points: int)
where
  reads(nodes.{x, k_coeff, l_x, r_x, l_k_coeff, r_k_coeff, lb , ub}),
  writes(nodes.{a, b, c})
do
  --C.printf('in set_abc\n')
  __demand(__openmp)
  for node in nodes do
    ----C.printf('in for loop\n')
    var i = (int) (int1d(node))
    ----C.printf("i: %d\n", i)
    --if i == i_lb then
    --  node.a = (0.5*(node.k_coeff + lb_k_coeff) / (node.x-lb_x)) / (0.5*(nodes[i+1].x - lb_x));
    --  node.b = ((-1.0)*0.5*(nodes[i+1].k_coeff + node.k_coeff) / (nodes[i+1].x - node.x) + (-1.0)*0.5*(node.k_coeff + lb_k_coeff) / (node.x - lb_x)) / (0.5*(nodes[i+1].x - lb_x));
    --  node.c = (0.5*(nodes[i+1].k_coeff + node.k_coeff) / (nodes[i+1].x - node.x)) / (0.5*(nodes[i+1].x - lb_x)); 
    --elseif i == i_ub then
    --  node.a = (0.5*(node.k_coeff + nodes[i-1].k_coeff) / (node.x-nodes[i-1].x)) / (0.5*(ub_x - nodes[i-1].x));
    --  node.b = ((-1.0)*0.5*(ub_k_coeff + node.k_coeff) / (ub_x - node.x) + (-1.0)*0.5*(node.k_coeff + nodes[i-1].k_coeff) / (node.x - nodes[i-1].x)) / (0.5*(ub_x - nodes[i-1].x));
    --  node.c = (0.5*(ub_k_coeff + node.k_coeff) / (ub_x - node.x)) / (0.5*(ub_x - nodes[i-1].x)); 
    --else
    --  node.a = (0.5*(node.k_coeff + nodes[i-1].k_coeff) / (node.x-nodes[i-1].x)) / (0.5*(nodes[i+1].x - nodes[i-1].x));
    --  node.b = ((-1.0)*0.5*(nodes[i+1].k_coeff + node.k_coeff) / (nodes[i+1].x - node.x) + (-1.0)*0.5*(node.k_coeff + nodes[i-1].k_coeff) / (node.x - nodes[i-1].x)) / (0.5*(nodes[i+1].x - nodes[i-1].x));
    --  node.c = (0.5*(nodes[i+1].k_coeff + node.k_coeff) / (nodes[i+1].x - node.x)) / (0.5*(nodes[i+1].x - nodes[i-1].x)); 
    --end


    if node.lb == 1 then
      node.a = (0.5*(node.k_coeff + node.l_k_coeff) / (node.x-node.l_x)) / (0.5*(nodes[i+1].x - node.l_x));
      node.b = ((-1.0)*0.5*(nodes[i+1].k_coeff + node.k_coeff) / (nodes[i+1].x - node.x) + (-1.0)*0.5*(node.k_coeff + node.l_k_coeff) / (node.x - node.l_x)) / (0.5*(nodes[i+1].x - node.l_x));
      node.c = (0.5*(nodes[i+1].k_coeff + node.k_coeff) / (nodes[i+1].x - node.x)) / (0.5*(nodes[i+1].x - node.l_x));
    elseif node.ub == 1 then
      node.a = (0.5*(node.k_coeff + nodes[i-1].k_coeff) / (node.x-nodes[i-1].x)) / (0.5*(node.r_x - nodes[i-1].x));
      node.b = ((-1.0)*0.5*(node.r_k_coeff + node.k_coeff) / (node.r_x - node.x) + (-1.0)*0.5*(node.k_coeff + nodes[i-1].k_coeff) / (node.x - nodes[i-1].x)) / (0.5*(node.r_x - nodes[i-1].x));
      node.c = (0.5*(node.r_k_coeff + node.k_coeff) / (node.r_x - node.x)) / (0.5*(node.r_x - nodes[i-1].x));
    elseif ((i ~= 0) and (i ~= num_grid_points-1)) then
      node.a = (0.5*(node.k_coeff + nodes[i-1].k_coeff) / (node.x-nodes[i-1].x)) / (0.5*(nodes[i+1].x - nodes[i-1].x));
      node.b = ((-1.0)*0.5*(nodes[i+1].k_coeff + node.k_coeff) / (nodes[i+1].x - node.x) + (-1.0)*0.5*(node.k_coeff + nodes[i-1].k_coeff) / (node.x - nodes[i-1].x)) / (0.5*(nodes[i+1].x - nodes[i-1].x));
      node.c = (0.5*(nodes[i+1].k_coeff + node.k_coeff) / (nodes[i+1].x - node.x)) / (0.5*(nodes[i+1].x - nodes[i-1].x));
    end
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

task calc_u(nodes: region(ispace(int1d),Node), num_grid_points: int)
where
  reads(nodes.{c_star, d_star}),
  reads writes(nodes.u)
do
  for i = num_grid_points-1,-1,-1 do
    if i == (num_grid_points-1) then
      nodes[i].u = nodes[i].d_star
    else
      nodes[i].u = nodes[i].d_star - nodes[i].c_star * nodes[i+1].u      
    end 
  end
end

-- change to reduces
__demand(__leaf, __parallel, __cuda)
task integrate_u(nodes: region(ispace(int1d), Node), grid_spacing: double) : double
where
  reads(nodes.u)
do
  var sum = 0.0;
  for node in nodes do
    sum += node.u*grid_spacing
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
  C.printf('\n')
  var args = regentlib.c.legion_runtime_get_input_args()

  var num_grid_points = 0
  var num_GPUs = 0
  for i = 1, args.argc do
    if C.strcmp(args.argv[i], '-num_grid_points') == 0 and i < args.argc-1 then
      num_grid_points = C.atoi(args.argv[i+1])
    elseif C.strcmp(args.argv[i], '-num_GPUs') == 0 and i < args.argc-1 then
      num_GPUs = C.atoi(args.argv[i+1])
    end
  end
  
  var grid_spacing = DOMAIN_LENGTH/(num_grid_points-1.0)

  var index_space = ispace(int1d, {num_grid_points})
  var nodes = region(index_space, Node) 

  -- this is floor; would be better to round
  var partition_size = (int) ((double) (num_grid_points)/num_GPUs)

  var colors = ispace(int1d, {num_GPUs})
  var coloring = regentlib.c.legion_domain_point_coloring_create()
  for i = 0, num_GPUs do
    var range = rect1d{0, 1} --initializing range
    if i == num_GPUs-1 then
      range = rect1d{partition_size*i, num_grid_points-1}
    else
      range = rect1d{partition_size*i, partition_size*(i+1)-1}
    end
    regentlib.c.legion_domain_point_coloring_color_domain(coloring, [int1d](i), range)
  end
  var nodes_partition = partition(disjoint, nodes, coloring, colors)
  regentlib.c.legion_domain_point_coloring_destroy(coloring)

  -- check partition correct
  for c in colors do
    C.printf('color %d \n', c)
    for node in nodes_partition[c] do
      C.printf('    node %d \n', int1d(node))
    end
  end

  fill(nodes.d, F)
  fill(nodes.lb, 0)
  fill(nodes.ub, 0)

  set_xk(nodes, grid_spacing)

  __fence(__execution, __block)
  C.printf("after set_xk\n")
  __fence(__execution, __block)

  --var lb_is: double[MAX_NUM_GPUS]
  --var ub_is: double[MAX_NUM_GPUS]
  --var lb_k_coeffs: double[MAX_NUM_GPUS]
  --var ub_k_coeffs: double[MAX_NUM_GPUS]
  --var lb_xs: double[MAX_NUM_GPUS]
  --var ub_xs: double[MAX_NUM_GPUS]

  -- make this a CUDA task if works
  --for i=0,num_GPUs do
  --  var i_lb = partition_size*i-1
  --  var i_ub = partition_size*(i+1)
  --  var node_lb = nodes[i_lb]
  --  var node_ub = nodes[i_ub]
  --  lb_is[i] = i_lb
  --  ub_is[i] = i_ub
  --  lb_k_coeffs[i] = node_lb.k_coeff
  --  ub_k_coeffs[i] = node_ub.k_coeff
  --  lb_xs[i] = node_lb.x
  --  ub_xs[i] = node_ub.x
  --end
  for i =0,num_GPUs do
    if i == 0 then
      var i_r = partition_size-1
      nodes[i_r].r_x = nodes[i_r+1].x
      nodes[i_r].r_k_coeff = nodes[i_r+1].k_coeff
      nodes[i_r].ub = 1
      C.printf("rb i: %d", i_r)
    elseif i == num_GPUs-1 then
      var i_l = partition_size*(num_GPUs-1)
      nodes[i_l].l_x = nodes[i_l-1].x
      nodes[i_l].l_k_coeff = nodes[i_l-1].k_coeff
      nodes[i_l].lb = 1
      C.printf("lb i: %d", i_l)
    else
      var i_r = (i+1)*partition_size-1
      nodes[i_r].r_x = nodes[i_r+1].x
      nodes[i_r].r_k_coeff = nodes[i_r+1].k_coeff
      nodes[i_r].ub = 1

      var i_l = partition_size*i
      nodes[i_l].l_x = nodes[i_l-1].x
      nodes[i_l].l_k_coeff = nodes[i_l-1].k_coeff
      nodes[i_l].lb = 1

      C.printf("rb i: %d", i_r)
      C.printf("lb i: %d", i_l)
    end
  end

 
  __fence(__execution, __block)
  C.printf("before set_abc\n") 
  __fence(__execution, __block)

  for c in colors do
    var i = (int) (c)
    --__fence(__execution, __block)
    --C.printf("c: %d\n", c)
    set_abc(nodes_partition[c], num_grid_points)
    --set_abc(nodes_partition[c], lb_is[i], ub_is[i], lb_k_coeffs[i], ub_k_coeffs[i], lb_xs[i], ub_xs[i])
    --__fence(__execution, __block)
  end  

  -- correct boundary nodes 
  nodes[0].a = 0.0;
  nodes[0].b = 1.0;
  nodes[0].c = 0.0;
  nodes[0].d = U_0;
  nodes[num_grid_points-1].a = 0.0;
  nodes[num_grid_points-1].b = 1.0;
  nodes[num_grid_points-1].c = 0.0;
  nodes[num_grid_points-1].d = U_1;

  __fence(__execution, __block)
  C.printf("after set_abc\n")
  __fence(__execution, __block)
 
  set_cd_star(nodes)

  __fence(__execution, __block)
  C.printf("after set_cd_star\n")
  __fence(__execution, __block)

  calc_u(nodes, num_grid_points)

  __fence(__execution, __block)
  C.printf("after calc_u\n")
  __fence(__execution, __block)

  __fence(__execution, __block)
  C.printf('x, k_coeff, a, b, c, d, c_star, d_star, u\n')
  for node in nodes do
    C.printf('%f %f %f %f %f %f %f %f %f \n', node.x, node.k_coeff, node.a, node.b, node.c, node.d, node.c_star, node.d_star, node.u)
  end
  __fence(__execution, __block)

  var sum = integrate_u(nodes, grid_spacing)

  __fence(__execution, __block)
  C.printf('sum = %f \n', sum)
  __fence(__execution, __block)
end

regentlib.saveobj(main, "diffusion.o")
