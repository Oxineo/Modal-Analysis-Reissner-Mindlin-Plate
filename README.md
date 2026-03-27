Reissner Mindlin Plate : Modal Analysis with Fenicsx 
=================

First of all, this work is base on Jeremie Bleyer's work which can be found with this link : [Fenicsx Tour : Reissner-Mindlin plates ](https://bleyerj.github.io/comet-fenicsx/intro/plates/plates.html)

Governor Equation
===============

Generalized strain
---------------

- Courbure : $ \chi = \nabla^S \phi  $
- Déformation de cisaillement : $ \gamma = \nabla u - \phi $

Generalized stresses
----------------

- Moment lié à la courbure : $ \textbf{M} $
- Force de cisaillement : $ \textbf{V} $ 

Formulation Forte 
----------------

 - $ V_{ \beta , \beta } = M \ddot U_3 $
 - $ M_{ \alpha \beta , \beta } - V_\alpha = I \ddot \phi _\alpha $
