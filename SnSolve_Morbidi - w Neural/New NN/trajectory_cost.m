    %MEAM517
%Greg Campbell & Spencer Folk
%Rework of Morbidi 2016 Paper

%NOTE - 11/8 - g and dG should now be correct.

function [g,dG] = trajectory_cost(z)
%Cost(z) computes the cost and cost jacobian - Matt Halm's base code.
%   @param z: decision variable (column) vector containing the xi and u_i
%   @param N: number of sample points; scalar
%   @param nx: dimension of state vector, x; scalar
%   @param nu: dimension of input vector, u; scalar
%   @param dt: \Delta t, the inter-sample interval duration; scalar

%   @output g: total accrued cost; scalar
%   @output dG: gradient of total accrued cost; nz by 1 vector

%Note: nx = 16, nu = 4, dt = input, N = 100 (he specifies 100 intervals
%in section IV A),
    consts = constants();
    
    costtype = consts.costtype;
    %Choose function based on cost type
        if costtype == 'NeuralNet'
            [g,dG] = NeuralCost(z,consts);
        elseif costtype == 'Trapezoid'
            [g,dG] = TrapCost(z,consts);
        elseif costtype == 'Rectangle'
            [g,dG] = RectCost(z,consts);
        elseif costtype == 'Simple'
            [g,dG] = SimpleCost(z,consts);
        end

end

function [g,dG] = SimpleCost(z,consts)

    N = consts.N;
    nx = consts.nx;
    nu = consts.nu;
    dt = consts.dt;
    g = 0;
    dG = zeros(N*(nx + nu),1);
    for i=1:N
        g_i = sum(ui.^2);
        g = g+g_i;
    end

end

function [g,dG] = TrapCost(z,consts)
    N = consts.N;
    nx = consts.nx;
    nu = consts.nu;
    dt = consts.dt;
    g = 0;
    dG = zeros(N*(nx + nu),1);
    for i=1:(N-1)
       
        [x_i_inds, u_i_inds] = sample_indices(i);
        [x_ip1_inds, u_ip1_inds] = sample_indices(i+1);
        
        xi = z(x_i_inds);
        xip1 = z(x_ip1_inds);
        ui = z(u_i_inds);
        uip1 = z(u_ip1_inds);
        
        %Input equation       
            g_i = 0;

            g_i = g_i + (Energy1(xi(13:16),consts)+Energy1(xip1(13:16),consts))*dt/2;
            g_i = g_i + (Energy2(ui(1:4),consts)+Energy2(uip1(1:4),consts))*dt/2;    

            %Indiviual dG components based on partial derivatives
            %Note - x1-12 are not included in g and dG w.r.t. each of them is 0

            dG_xi = zeros(nx,1);
            dG_xip1 = zeros(nx,1);

            %Partial with respect to each state is identical x13-16
            dG_xi(13:16) = dt/2*(dEnergy1(xi(13:16),consts));
            dG_xip1(13:16) = dt/2*(dEnergy1(xip1(13:16),consts));
        
            %Partial with respect to each input is identical
            dG_ui = dt*consts.c7*ui;
            dG_uip1 = dt*consts.c7*uip1;

            %Update g and dG with added pieces from g_i and dG_i
            g = g+g_i;
            dG(u_i_inds) = dG(u_i_inds) + dG_ui;
            dG(u_ip1_inds) = dG_uip1;
            dG(x_i_inds) = dG(x_i_inds) + dG_xi;
            dG(x_ip1_inds) = dG_xip1;
    end

end

function [g,dG] = RectCost(z,consts)
    N = consts.N;
    nx = consts.nx;
    nu = consts.nu;
    dt = consts.dt;
    
    g = 0;
    dG = zeros(N*(nx + nu),1);
    
    for i=1:N
        [x_i_inds, u_i_inds] = sample_indices(i);
        
        xi = z(x_i_inds);
        ui = z(u_i_inds);
        
        %Input equation
            g_i = 0;

            g_i = dt*(sum(consts.c1 + consts.c2*xi(13:16) + consts.c3*xi(13:16).^2 ...
                + consts.c4*xi(13:16).^3 + consts.c5*xi(13:16).^4) + sum(ui.^2));


            %Indiviual dG components based on partial derivatives
            %Note - x1-12 are not included in g and dG w.r.t. each of them is 0

            dG_xi = zeros(nx,1);

            %Partial with respect to each state is identical x13-16
            dG_xi(13:16) = (consts.c2 + 2*consts.c3*xi(13:16) + ...
                3*consts.c4*xi(13:16).^2 + 4*consts.c5*xi(13:16).^3)*dt;

            %Partial with respect to each input is identical
            dG_ui = dt*consts.c7*ui;

            %Update g and dG with added pieces from g_i and dG_i
            g = g+g_i;
            dG(u_i_inds) = dG(u_i_inds) + dG_ui;
            dG(x_i_inds) = dG(x_i_inds) + dG_xi;
    end

end

function [g,dG] = NeuralCost(z,consts)
    N = consts.N;
    nx = consts.nx;
    nu = consts.nu;
    dt = consts.dt;
    g = 0;
    dG = zeros(N*(nx + nu),1);
    %Weight Variables
        Theta1 = load("optimized_Theta1.txt");
        Theta2 = load("optimized_Theta2.txt");
    
    %create an Nx16 matrix of 'states' used by Neural Network
    X = zeros(N,16);
 
    for i = 1:N
 
        [x_i_inds, u_i_inds] = sample_indices(i);
 
        xi = z(x_i_inds);
        ui = z(u_i_inds);
 
        X0 = zeros(1,16);
        
        X0(1) = norm([xi(2), xi(4)]);
        X0(2) = xi(6);
        X0(3) = xi(7);
        X0(4) = xi(9);
        X0(5) = xi(11);
        X0(6:9) = xi(13:16);
        X0(10) = xi(8);
        X0(11) = xi(10);
        X0(12) = xi(12);
        X0(13:16) = ui(1:4);
 
        X(i,:) = X0;
    end
        
    [~, mu, sigma] = featureNormalize(X);
%     norm_z = z;
        
    for i=1:N
        
        [x_i_inds, u_i_inds] = sample_indices(i);
        
        xi = z(x_i_inds);
        ui = z(u_i_inds);
        zi = [xi;ui];
        
        NeuralFun = @(zi) nnPredict(Theta1,Theta2,zi,sigma,mu);

        g_i = NeuralFun(zi)*dt;
        g = g + g_i;                

              % use numerical derivatives to compute dG
              % dG = [dh/dx0 dh/du0 dh/dx1 dh/du1]
              % where the partial derivatives are written (dh/dx0)_ij = dh_i/dx0_j
              delta = 1e-8;
              dG = zeros(N*(nx + nu),1);
              for j=1:numel(xi)
                  dx = zeros(numel(xi)+numel(ui),1);
                  dx(j) = delta;
                  dGx_i_j = NeuralFun(zi + dx) - g_i;
                  dG(x_i_inds(j)) = dGx_i_j/delta;
              end

              for j=1:numel(ui)
                  du = zeros(numel(xi)+numel(ui),1);
                  du(numel(xi)+j) = delta;
                  dGu_i_j = NeuralFun(zi + du) - g_i;
                  dG(u_i_inds(j)) = dGu_i_j/delta;
              end
    end
end

function [Er1] = Energy1(xki,constants)
    %Takes in [1,1] xki (all state values in relevant state
    %Takes in structure constants containing c1-c9
    %Gives back scalar Er1, corresponding to the portion of energy Er
    %corresponding to the current state
    Er1 = sum(constants.c1+constants.c2*xki+constants.c3*xki.^2+constants.c4*xki.^3+constants.c5*xki.^4);
end

function [dEr1_dxk] = dEnergy1 (xki,constants)
    %Takes in [1,1] xki (all state values in relevant state
    %Takes in structure constants containing c1-c9
    %Gives back scalar dEr1_dxk, corresponding to the gradient w.r.t. xk.
    %Note that this is the partial derivative of the above "Energy1"
    dEr1_dxk = (constants.c2+2*constants.c3*xki+3*constants.c4*xki.^2+4*constants.c5*xki.^3);
end

function [Er2] = Energy2(uji,constants)
    %Takes in [1,1] uji (all state values in relevant state
    %Takes in structure constants containing c1-c9
    %Gives back scalar Er2, corresponding to the portion of energy Er
    %corresponding to the input
    Er2 = sum(constants.c7*uji.^2);
end