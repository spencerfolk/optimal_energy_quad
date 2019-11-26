%MEAM517
%Greg Campbell & Spencer Folk
%Rework of Morbidi 2016 Paper

%NOTE - 11/8 - g and dG should now be correct.

function [g,dG] = trajectory_cost(z, N, nx, nu, dt)
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
consts = constants();
g = 0;
dG = zeros(N*(nx + nu),1);

for i=1:(N-1)
   
    [x_i_inds, u_i_inds] = sample_indices(i, nx, nu);
%     [x_ip1_inds, u_ip1_inds] = sample_indices(i+1, nx, nu);
   
    xi = z(x_i_inds);
%     xip1 = z(x_ip1_inds);
    ui = z(u_i_inds);
%     uip1 = z(u_ip1_inds);
   
    %Input equation
    g_i = 0;
   
    %     for k = 13:16
    % %         g_i = g_i + (Energy1(xi(k),consts)+Energy1(xip1(k),consts))*dt/2;
    %         g_i = g_i + (Energy1(xi(k),consts))*dt;
    %     end
    %     for j = 1:4
    % %         g_i = g_i + (Energy2(ui(j),consts)+Energy2(uip1(j),consts))*dt/2;
    %         g_i = g_i + Energy2(ui(j),consts)*dt;
    %     end
   
    g_i = dt*(sum(consts.c1 + consts.c2*xi(13:16) + consts.c3*xi(13:16).^2 ...
        + consts.c4*xi(13:16).^3 + consts.c5*xi(13:16).^4) + sum(ui.^2));
   
   
    %Indiviual dG components based on partial derivatives
    %Note - x1-12 are not included in g and dG w.r.t. each of them is 0
   
    dG_xi = zeros(nx,1);
%     dG_xip1 = zeros(nx,1);
   
    %Partial with respect to each state is identical x13-16
%     for k = 13:16
%         dG_xi(k) = dt/2*(dEnergy1(xi(k),consts));
% %         dG_xip1(k) = dt/2*(dEnergy1(xip1(k),consts));
%     end
    dG_xi(13:16) = (consts.c2 + 2*consts.c3*xi(13:16) + ...
        3*consts.c4*xi(13:16).^2 + 4*consts.c5*xi(13:16).^3)*dt;

    %Partial with respect to each input is identical
%     dG_ui = dt*consts.c7*ui;
    dG_ui = dt*consts.c7*ui;
%     dG_uip1 = dt*consts.c7*uip1;
   
    %Update g and dG with added pieces from g_i and dG_i
    g = g+g_i;
    dG(u_i_inds) = dG(u_i_inds) + dG_ui;
%     dG(u_ip1_inds) = dG_uip1;
    dG(x_i_inds) = dG(x_i_inds) + dG_xi;
%     dG(x_ip1_inds) = dG_xip1;
   
end

end

function [Er1] = Energy1(xki,constants)
%Takes in [1,1] xki (all state values in relevant state
%Takes in structure constants containing c1-c9
%Gives back scalar Er1, corresponding to the portion of energy Er
%corresponding to the current state
Er1 = (constants.c1+constants.c2*xki+constants.c3*xki^2+constants.c4*xki^3+constants.c5*xki^4);
end

function [dEr1_dxk] = dEnergy1 (xki,constants)
%Takes in [1,1] xki (all state values in relevant state
%Takes in structure constants containing c1-c9
%Gives back scalar dEr1_dxk, corresponding to the gradient w.r.t. xk.
%Note that this is the partial derivative of the above "Energy1"
dEr1_dxk = constants.c2 + 2*constants.c3*xki+3*constants.c4*xki^2+4*constants.c5*xki^3;
end

function [Er2] = Energy2(uji,constants)
%Takes in [1,1] uji (all state values in relevant state
%Takes in structure constants containing c1-c9
%Gives back scalar Er2, corresponding to the portion of energy Er
%corresponding to the input
Er2 = constants.c7*uji^2;
end