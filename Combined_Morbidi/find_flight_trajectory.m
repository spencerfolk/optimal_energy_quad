
%MEAM517
%Greg Campbell & Spencer Folk
%Rework of Morbidi 2016 Paper

%This function runs fmincon on the trajectory optimization problem.

% X = fmincon(PROBLEM) finds the minimum for PROBLEM. PROBLEM is a
%     structure with the function FUN in PROBLEM.objective, the start point
%     in PROBLEM.x0, the linear inequality constraints in PROBLEM.Aineq
%     and PROBLEM.bineq, the linear equality constraints in PROBLEM.Aeq and
%     PROBLEM.beq, the lower bounds in PROBLEM.lb, the upper bounds in 
%     PROBLEM.ub, the nonlinear constraint function in PROBLEM.nonlcon, the
%     options structure in PROBLEM.options, and solver name 'fmincon' in
%     PROBLEM.solver. Use this syntax to solve at the command line a problem 
%     exported from OPTIMTOOL.


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
        ub(x_i_inds(13:16)) = w_max;
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
