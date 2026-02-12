clear; clc;

%% O objetivo do exemplo eh mostrar a propriedade de diagonalizacao
%% da matriz de covariancia dos dados. Em outras palavras, verificar
%% se a matriz de covariancia dos dados transformados Z eh diagonal
%% e se as variancias sao iguais aos autovalores da matriz de covariancia
%% Cx (dados originais).

%% Autor: Guilherme Barreto
%% Data: 17/05/2023

clear; clc; close all

pkg load statistics

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gera dados gaussianos com atributos nao-correlacionados  %%
%% a partir de dados com atributos descorrelacionados       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m1=5;  % Media teorica do atributo 1
sig1 = 1; % Desvio-padrao teorico do atributo 1

m2=-5; % Media teorica do atributo 2
sig2 = sig1; % Desvio-padrao teorico do atributo 2

m3=0; % Media teorica do atributo 3
sig3 = sig1; % Desvio-padrao teorico do atributo 3

N = 50000;   % Quantidade de observacoes geradas de cada atributo
X1=normrnd(m1,sig1,1,N);
X2=normrnd(m2,sig2,1,N);
X3=normrnd(m3,sig3,1,N);

Xu=[X1; X2; X3];  % Agrupa dados dos atributos em uma unica matriz

% Matriz desejada para os dados
Cd=[1 1.8 -0.9;1.8 4 0.6;-0.9 0.6 9];

R=chol(Cd);  % Decomposicao de Cholesky da matriz Cd

Xc=R'*Xu;  % Gera dados com atributos correlacionados com a matriz COV desejada

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Aplicacao de PCA aos dados correlacionados gerados no  %%
%% procedimento anterior.                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Cx = cov(Xc');  % Estima a matriz de covariancia dos dados simulados
[V L]=eig(Cx); L=diag(L);
[L I]=sort(L,'descend'); V=V(:,I);
VEi=100*L/sum(L);  % Variancia explicada pelo i-esimo autovalor
VEq=100*cumsum(L)/sum(L);   % Variancia explicada pelos q primeiros autovalores

figure; plot(VEq,'r-','linewidth',2);
xlabel('Numero de autovalores principais (q)'); ylabel('Variancia Explicada');
set(gca, "fontsize", 14)
print screeplot.eps -color -deps

%[V2 L2 VEi2] = pcacov(Cx);  % Calcula autovalores e autovetores da matriz Cx
%VEq2=cumsum(VEi);

[U3 L3 V3]=svd(Cx); %%%% PCA a partir da SVD %%%%%%

Q = V';  % Monta matriz de transformacao (sem reducao de dimensionalidade)
% Q=V2';
% Q=V3';

Z = Q*Xc;  % Gera dados via PCA (descorrelaciona matriz dos dados)

Cz = cov(Z');  % Matriz de covariancia empirica dos dados transformados via PCA

figure;
subplot(1,2,1)
plot(Xc(1,1:5000),Xc(2,1:5000),'ro','linewidth',2);
xlabel('Atributo X1'); ylabel('Atributo X2');
title('Dados Originais');
set(gca, "fontsize", 14)

subplot(1,2,2)
plot(Z(1,1:5000),Z(2,1:5000),'bo','linewidth',2);
xlabel('Atributo Z1'); ylabel('Atributo Z2');
title('Dados Transformados');
set(gca, "fontsize", 14)

print pcademo.eps -color -deps

%%%% Reconstrucao dos dados originais
Xr = Q'*Z;  % Conjunto de dados original recuperado a partir de Z

E=Xc-Xr;   % Vetor erro de reconstrucao
NormaE2 = norm(E,'fro')^2  % Norma de Frobenius para calcular o erro.

E=E(:);  % Vetoriza matriz de erro
SSE=sum(E.^2)  % Soma dos erros quadraticos de reconstrucao





