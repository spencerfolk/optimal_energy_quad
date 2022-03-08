%MEAM517
%Greg Campbell & Spencer Folk
%Rework of Morbidi 2016 Paper

%Note: Page numbers will be based off of PDF w/ title page

%% Constants for Phantom 2 Quadrotor - Page 5/8 (table)

K_v = 920; %rpm/V
n_B = 2;
rho = 1.225; %kg/m^3
m_B = 0.0055; %kg
m = 1.3; %kg
T_f = 4e-2; %N*m
r = 0.12; %m
l = 0.175; %m
D_f = 2e-4; %N*m*s/rad
eps = 0.004; %m
I_x = 0.081; %kg*m^2
I_y = 0.081; %kg*m^2
I_z = 0.142; %kg*m^2
R = 0.2; % Ohm
C_T = 0.0048;
J_m = 2.9e-6; %kg*m^2
C_Q = 2.3515e-4;
w_max = 1047.197; %rad/s
r_rot = 0.014; %m
m_rot = 0.025; %kg
g = 9.8066; %m/s^2

w_h = 912.109; %rad/s - page 6 under (13)

%Calculated constants
K_E = 9.5493/K_v; %V*s/rad
K_T = K_E; %per page 3.
J_L = 1/4*n_B*m_B*(r-eps)^2;
A = pi*r^2;
k_b = C_T*rho*A*r^2;
k_tao = C_Q*rho*A*r^3;
J = J_m+J_L;

%Constants
constants.c1 = R*T_f^2/K_T^2;
constants.c2 = T_f/K_T*(2*R*D_f/K_T+K_E);
constants.c3 = D_f/K_T*(R*D_f/K_T+K_E)+2*R*T_f*k_tao/K_T^2;
constants.c4 = k_tao/K_T*(2*R*D_f/K_T+K_E);
constants.c5 = R*k_tao^2/K_T^2;
constants.c6 = 2*R*J*T_f/K_T^2;
constants.c7 = R*J^2/K_T^2;
constants.c8 = J/K_T*(2*R*D_f/K_T+K_E);
constants.c9 = 2*R*J*k_tao/K_T^2;

%% Simplified Energy Model

%NOTE - 11/8 - g and dG should now be correct.

function [g,dG] = cost(z, N, nx, nu, dt)
%Cost(z) computes the cost and cost jacobian - Matt Halm's base code.
%   @param z: decision variable (column) vector containing the x_i and u_i
%   @param N: number of sample points; scalar
%   @param nx: dimension of state vector, x; scalar
%   @param nu: dimension of input vector, u; scalar
%   @param dt: \Delta t, the inter-sample interval duration; scalar

%   @output g: total accrued cost; scalar
%   @output dG: gradient of total accrued cost; nz by 1 vector

%Note: nx = 16, nu = 4, dt = input, N = 100 (he specifies 100 intervals
%in section IV A),
    g = 0;
    dG = zeros(N*(nx + nu),1);

    for i=1:(N-1)
       
        [x_i_inds, u_i_inds] = sample_indices(i, nx, nu);
        [x_ip1_inds, u_ip1_inds] = sample_indices(i+1, nx, nu);
        
        xi = z(x_i_inds);
        xip1 = z(x_ip1_inds);
        ui = z(u_i_inds);
        uip1 = z(u_ip1_inds);
        
        %Input equation       
        g_i = 0;
                
    for k = 13:16
        g_i = g_i + (Energy1(xi(k),constants)+Energy1(xip1(k),constants))*dt/2;
    end
    for j = 1:4
        g_i = g_i + (Energy2(ui(j),constants)+Energy2(uip1(j),constants))*dt/2;
    end    
        
        %Indiviual dG components based on partial derivatives
        %Note - x1-12 are not included in g and dG w.r.t. each of them is 0
        
        dG_xi = zeros(nx,1);
        dG_xip1 = zeros(nx,1);
        
        %Partial with respect to each state is identical x13-16
        for k = 13:16    
            dG_xi(k) = dt/2*(dEnergy1(xi(k)));
            dG_xip1(k) = dt/2*(dEnergy1(xip1(k)));
        end
        
        %Partial with respect to each input is identical
        dG_ui = dt*constants.c7*ui;
        dG_uip1 = dt*constants.c7*uip1;
        
        %Update g and dG with added pieces from g_i and dG_i
        g = g+g_i;
        dG(u_i_inds) = dG(u_i_inds) + dG_ui;
        dG(u_ip1_inds) = dG_uip1;
        dG(x_i_inds) = dG(x_i_inds) + dG_xi;
        dG(x_ip1_inds) = dG_xip1;

    end

end

function [Er1] = Energy1(xki,constants)
    %Takes in [1,1] xki (all state values in relevant state
    %Takes in structure constants containing c1-c9
    %Gives back scalar Er1, corresponding to the portion of energy Er
    %corresponding to the current state
%     Er1 = (constants.c1+constants.c2*xki+constants.c3*xki^2+constants.c4*xki^3+constants.c5*xki^4);
    %Greg Bullshitting 3/8/22 - to be deleted:
    Er1 = (constants.c2+constants.c1*xki+constants.c3*xki^2+constants.c4*xki^3+constants.c5*xki^4);
end

function [dEr1_dxk] = dEnergy1 (xki,constants)
    %Takes in [1,1] xki (all state values in relevant state
    %Takes in structure constants containing c1-c9
    %Gives back scalar dEr1_dxk, corresponding to the gradient w.r.t. xk.
    %Note that this is the partial derivative of the above "Energy1"
    dEr1_dxk = constants.c2*2*constants.c3*xki+3*constants.c4*xki^2+4*constants.c5*xki^3;
end

function [Er2] = Energy2(uji,constants)
    %Takes in [1,1] uji (all state values in relevant state
    %Takes in structure constants containing c1-c9
    %Gives back scalar Er2, corresponding to the portion of energy Er
    %corresponding to the input
    Er2 = constants.c7*uji^2;
end

%NOTE - 11/8: the following is updated with the EXCEPTION of constraints in
%'all_contraints'

function [z, Aeq, beq, lb, ub, z0] = find_flight_trajectory(x_0, x_f, N, dt)
%FIND_SWINGUP_TRAJECTORY(x_0, x_f, N, dt) executes a direct collocation
%optimization program to find an input sequence to drive the cartpole
%system from x_0 to x_f.
%
%   @param x_0: the state at the start of the trajectory; n_x by 1 vector
%   @param x_f: the state at the emd of the trajectory; n_x by 1 vector
%
%   @output z: decision variable vector containing the x_i and u_i
%   @output Aeq: matrix from linear constrant Aeq z = beq
%   @output beq: (column) vector from linear constrant Aeq z = beq
%   @output lb: lower bound of constraint lb <= z <= ub; n_z by 1 vector
%   @output ub: upper bound of constraint lb <= z <= ub; n_z by 1 vector
%   @output z0: initial guess for z; n_z by 1 vector
    nx = 16;
    nu = 4;

    % TODO: Add constraints to Aeq, beq to enforce starting at x_0 and
    % ending at x_f
    Aeq = zeros(2*nx, N * (nx + nu));
    beq = zeros(2*nx, 1);
    
    %Fill in Aeq in area of x1 and xN
    [x_1_inds,~] = sample_indices(1, nx, nu);
    [x_N_inds,~] = sample_indices(N, nx, nu);
    Aeq(1:nx,x_1_inds) = eye(nx);
    Aeq(nx+1:2*nx,x_N_inds) = eye(nx);
    
    %Fill in beq with given inital and final conditions
    beq = [x_0;x_f];
    
    w_max = 1047.197; %rad/s

    % TODO: Add bounding box constraints u \in [-M,M]^nu
    lb = -inf(N * (nx + nu),1);
    ub = inf(N * (nx + nu),1);
    
    for i=1:N
        
        % Add bounding box constraints for u_i
        [x_i_inds,~] = sample_indices(i, nx, nu);      
        lb(x_i_inds(13:16)) = 0;
        ub(u_i_inds(13:16)) = w_max;
    end

    % TODO: make initial guess for z
    z0 = zeros(N * (nx + nu), 1);
    
    % Create a spline (linear) from x_0 to x_f and discretize over dt
    t = [0 N*dt];
    x = [x_0 x_f];
    for j=1:nx
        x_spline = spline(t, x(j,:));
        t_in = 0:dt:N*dt;
        Xdisc(j,:) = ppval(x_spline,t_in);
    end

    for i=1:N
        % TODO: make initial guess for ith sample
        [x_i_inds,~] = sample_indices(i, nx, nu);
        
        z0(x_i_inds) = Xdisc(:,i);
    end

    options = optimoptions('fmincon','SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,'Display','iter');
    problem.objective = @(z) trajectory_cost(z, N, nx, nu, dt);


    problem.x0 = z0;
    problem.options = options;
    problem.nonlcon = @(z) all_constraints(z, N, nx, nu, dt);
    problem.solver = 'fmincon';
    problem.Aeq = Aeq;
    problem.beq = beq;
    problem.lb = lb;
    problem.ub = ub;

    z = fmincon(problem);
end

function [c, ceq, dC, dCeq] = all_constraints(z, N, nx, nu, dt)

    [ceq, dCeq] = dynamics_constraints(z, N, nx, nu, dt);

    c = zeros(0,1);
    dC = zeros(0,numel(z));

    dC = sparse(dC)';
    dCeq = sparse(dCeq)';
end

%Note - 11/8: the following two functions have been unupdated from
%Homework 5. They must be updated to include the dynamics of our system.

function [h,dH] = dynamics_constraints(z, N, nx, nu, dt)
%DYNAMICS_CONSTRAINTS(z) compiles the dynamics constraints generated by
%dynamics_constraint_with_derivative.
%   @param z: decision variable (column) vector containing the x_i and u_i
%   @param N: number of sample points; scalar
%   @param nx: dimension of state vector, x; scalar
%   @param nu: dimension of input vector, u; scalar
%   @param dt: \Delta t, the inter-sample interval duration; scalar

%   @output h: compiled h_i from dynamics_constraint_with_derivative;
%   (N-1)*nx by 1 vector
%   @output dH_i: compiled dH_i from dynamics_constraint_with_derivative;
%   (N-1)*nx by nz matrix

    h = zeros((N-1)*nx, 1);
    dH = zeros((N-1)*nx, N*(nx + nu));

    for i=1:(N-1)

        % TODO: call dynamics_constraint_with_derivative ith sample
        %Manage indices and solve for x/u i and ip1
        [x_i_inds, u_i_inds] = sample_indices(i, nx, nu);
        [x_ip1_inds, u_ip1_inds] = sample_indices(i+1, nx, nu);
        x_i = z(x_i_inds);
        u_i = z(u_i_inds);
        x_ip1 = z(x_ip1_inds);
        u_ip1 = z(u_ip1_inds);
        %Call dynamics_constraint_with_derivative
        [h_i,dH_i] = dynamics_constraint_with_derivative(x_i, u_i, x_ip1, u_ip1, dt);

        % TODO fit h_i and dH_i into h and dH, respectively.
        h(x_i_inds) = h_i;
        dH(x_i_inds,[x_i_inds,u_i_inds,x_ip1_inds,u_ip1_inds]) = dH_i;

    end

end

function [h_i,dH_i] = dynamics_constraint_with_derivative(x_i, u_i, x_ip1, u_ip1, dt)
%DYNAMICS_CONSTRAINT_WITH_DERIVATIVE(x_i, u_i, x_ip1, u_ip1, dt) returns
%and computes the gradient of the vector constraint asssociated with
%dynamics_constraint(x_i, u_i, x_ip1, u_ip1, dt).
%Originally created by Matt Halm for MEAM517.
%
%   @param x_i: the state at the start of the interval; nx by 1 vector
%   @param u_i: the input at the start of the interval; nu by 1 vector
%   @param x_ip1: the state at the end of the interval; nx by 1 vector
%   @param u_ip1: the input at the end of the interval; nu by 1 vector
%   @param dt: \Delta t, the duration of the interval; scalar
%
%   @output h_i: constraint value from dynamics_constraint; nx by 1 vector
%   @output dH_i: jacobian of h_i w.r.t. [x_i; u_i; x_ip1; u_ip1]; nx by
%   (2nx + 2nu) matrix

    h_i = evaluate_dynamics_constraint(x_i, u_i, x_ip1, u_ip1, dt);
    if nargout > 1
      % use numerical derivatives to compute dH
      % dH = [dh/dx0 dh/du0 dh/dx1 dh/du1]
      % where the partial derivatives are written (dh/dx0)_ij = dh_i/dx0_j
      delta = 1e-8;
      dH_i = zeros(numel(x_i), 2*(numel(x_i)+numel(u_i)));
      for j=1:numel(x_i)
          dx = zeros(numel(x_i),1);
          dx(j) = delta;
          dHx_i_j = evaluate_dynamics_constraint(x_i + dx, u_i, x_ip1, u_ip1, dt) - h_i;
          dHx_ip1_j = evaluate_dynamics_constraint(x_i, u_i, x_ip1 + dx, u_ip1, dt) - h_i;
          dH_i(:,j) = dHx_i_j/delta;
          dH_i(:,j + numel(x_i) + numel(u_i)) = dHx_ip1_j/delta;
      end

      for j=1:numel(u_i)
          du = zeros(numel(u_i),1);
          du(j) = delta;
          dHu_i_j = evaluate_dynamics_constraint(x_i, u_i + du, x_ip1, u_ip1, dt) - h_i;
          dHu_ip1_j = evaluate_dynamics_constraint(x_i, u_i, x_ip1, u_ip1 + du, dt) - h_i;
          dH_i(:,j + numel(x_i)) = dHu_i_j/delta;
          dH_i(:,j + numel(x_i) + numel(u_i) + numel(x_ip1)) = dHu_ip1_j/delta;
      end
    end
end


function h_i = evaluate_dynamics_constraint(x_i, u_i, x_ip1, u_ip1, dt)
%EVALUATE_DYNAMICS_CONSTRAINT(x_i, u_i, x_ip1, u_ip1, dt) computes the
%   vector contstraint h_i(x_i, u_i, x_ip1, u_ip1) = 0 from Problem 2(c)
%
%   @param x_i: the state at the start of the interval; nx by 1 vector
%   @param u_i: the input at the start of the interval; nu by 1 vector
%   @param x_ip1: the state at the end of the interval; nx by 1 vector
%   @param u_ip1: the input at the end of the interval; nu by 1 vector
%   @param dt: \Delta t, the duration of the interval; scalar
%
%   @output h_i: quantity derived in Problem 2(c); nx by 1 vector
nx = numel(x_i);

h_i = zeros(nx,1);

% TODO: Calculate constraint value

%Impliment equtions solved in 3b and 3c
Si = 0.5*(x_i+x_ip1)-dt/8*(f(x_ip1,u_ip1)-f(x_i,u_i));
Sdi = 3/(2*dt)*(x_ip1-x_i)-1/4*(f(x_ip1,u_ip1)+f(x_i,u_i));
ri = 0.5*(u_i+u_ip1);

h_i = Sdi - f(Si,ri);

end

%{
X = fmincon(PROBLEM) finds the minimum for PROBLEM. PROBLEM is a
    structure with the function FUN in PROBLEM.objective, the start point
    in PROBLEM.x0, the linear inequality constraints in PROBLEM.Aineq
    and PROBLEM.bineq, the linear equality constraints in PROBLEM.Aeq and
    PROBLEM.beq, the lower bounds in PROBLEM.lb, the upper bounds in 
    PROBLEM.ub, the nonlinear constraint function in PROBLEM.nonlcon, the
    options structure in PROBLEM.options, and solver name 'fmincon' in
    PROBLEM.solver. Use this syntax to solve at the command line a problem 
    exported from OPTIMTOOL.
}%
