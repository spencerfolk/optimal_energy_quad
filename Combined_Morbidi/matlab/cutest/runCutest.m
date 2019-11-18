function runCutest ( probname )
% function runCutest ( probname )
%   A very simple m-file to run SNOPT on a problem from CUTEST in matlab
%

addpath([pwd,'/../'       ], '-end');
setenv('DYLD_LIBRARY_PATH', '/usr/local/bin:/opt/local/bin:');

cmd = ['runcutest --blas "" -p matlab -D ', probname ];
unix(cmd);

% Set up problem
prob   = cutest_setup();
name   = prob.name;

n      = prob.n;
m      = prob.m;
nF     = m + 1;

x0     = prob.x;

xl     = prob.bl(1:n);
xu     = prob.bu(1:n);

cl     = prob.cl(1:m);
cu     = prob.cu(1:m);


% Adjust bounds on linear constraints to
% take into account constants
%  1.  Find indices for finite bounds
%  2.  Find indices for linear constraints
%  3.  Subtract constant from bounds of linear constraints
x      = zeros(n,1);
c      = cutest_cons(x);

infBnd = 10^20;
linear = find( prob.linear(1:m) == 1 );
blBnd  = ( cl == -infBnd );
buBnd  = ( cu ==  infBnd );

indexL = ( linear > 0 & blBnd == 0 );
indexU = ( linear > 0 & buBnd == 0 );

cl([indexL]) = cl([indexL]) - c([indexL]);
cu([indexU]) = cu([indexU]) - c([indexU]);


% Objective row bounds
cl(nF) = -Inf;
cu(nF) =  Inf;

ObjRow = nF;
ObjAdd = 0;


xstate = zeros(n,1);
Fstate = zeros(nF,1);
xmul   = zeros(n,1);
Fmul   = zeros(nF,1);


% Set output
snprint( [probname, '.out'] );
snscreen on;

% Read spec file
info = snspec('cutest.spc');

% Solve the problem
[x,F,inform,xmul,Fmul] = snsolve (x0, xl, xu, xmul, xstate,...
                                  cl, cu, Fmul, Fstate, @cutestUserfun,...
				  ObjAdd,ObjRow);

% Close output streams
snprint off;
snscreen off;
snend;