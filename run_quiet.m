function varargout = run_quiet(fn)
% Execute fn() en avalant tout ce qui s'affiche dans la console : bannieres
% "Welcome" de la drtoolbox, traces "MSE of ... model" de l'autoencodeur,
% "Constructing neighborhood graph...", et warnings verbeux (matrice
% singuliere, issym). Seule la progression du script reste visible.
%
%   [a, b] = run_quiet(@() compute_mapping(X, 'Isomap', d));
    [~, varargout{1:nargout}] = evalc('fn()');
end
