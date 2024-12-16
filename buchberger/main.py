from sympy import symbols, lcm
from sympy.core.mul import Mul
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

def lcm_monomials(m1, m2):
    """
    Compute the least common multiple (LCM) of two monomials.
    """
    return tuple(max(e1, e2) for e1, e2 in zip(m1, m2))


def divide_polynomial_by_set(S, G):
    """
    Divide polynomial S by a set of polynomials G.
    
    Parameters:
    S (dict): Polynomial to divide, represented as a dictionary of monomials.
    G (list of dict): Set of polynomials, each represented as a dictionary.
    
    Returns:
    tuple: Quotients and remainder after division.
    """
    remainder = {}
    quotients = [dict() for _ in G]  # Initialize quotients for each g in G
    S_poly = S.copy()  # Working copy of the polynomial to divide

    while S_poly:
        # Get the leading term of the polynomial S
        lt_S = max(S_poly.keys(), key=lambda k: (sum(k), k))  # Max by degree, then lexicographically
        divisible = False

        for i, g in enumerate(G):
            if g:
                # Get the leading term of g
                lt_g = max(g.keys(), key=lambda k: (sum(k), k))
                
                # Check if the leading term of S is divisible by the leading term of g
                if all(e1 >= e2 for e1, e2 in zip(lt_S, lt_g)):
                    # Compute the quotient monomial
                    quotient_monomial = tuple(e1 - e2 for e1, e2 in zip(lt_S, lt_g))
                    quotient_coeff = S_poly[lt_S] / g[lt_g]

                    # Add the quotient to the respective quotient polynomial
                    quotients[i][quotient_monomial] = quotient_coeff

                    # Subtract from S_poly
                    for monomial, coeff in g.items():
                        new_monomial = tuple(q + m for q, m in zip(quotient_monomial, monomial))
                        S_poly[new_monomial] = S_poly.get(new_monomial, 0) - coeff * quotient_coeff
                        if S_poly[new_monomial] == 0:
                            del S_poly[new_monomial]  # Remove zero terms
                    
                    divisible = True
                    break

        if not divisible:
            # If no division occurred, add the leading term of S to the remainder
            remainder[lt_S] = S_poly.pop(lt_S)

    return quotients, remainder


def polynomial_to_string(poly_dict):
    """
    Convert a polynomial dictionary to a readable string format.
    
    Parameters:
    poly_dict (dict): Polynomial represented as a dictionary of monomials
    
    Returns:
    str: Human-readable string representation of the polynomial
    """
    if not poly_dict:
        return "0"
    
    terms = []
    for monomial, coeff in sorted(poly_dict.items(), key=lambda x: (-sum(x[0]), x[0]), reverse=True):
        if coeff == 0:
            continue
            
        # Handle coefficient
        term = ""
        if coeff != 1 or all(exp == 0 for exp in monomial):
            term = str(coeff)
            
        # Handle variables
        vars_part = []
        for i, exp in enumerate(monomial):
            var = ['x', 'y'][i]  # Add more variables if needed
            if exp > 0:
                if exp == 1:
                    vars_part.append(var)
                else:
                    vars_part.append(f"{var}^{exp}")
        
        term += "".join(vars_part)
        
        if coeff > 0 and terms:
            terms.append("+ " + term)
        elif coeff < 0:
            terms.append("- " + term.replace("-", ""))
        else:
            terms.append(term)
    
    return " ".join(terms)

def visualize_division(S, G):
    """
    Visualize the polynomial division process.
    
    Parameters:
    S (dict): Input polynomial
    G (list of dict): List of divisor polynomials
    """
    print("\n=== Gröbner Basis Division Visualization ===\n")
    
    print("Input polynomial:")
    print(f"S = {polynomial_to_string(S)}")
    
    print("\nDivisor polynomials:")
    for i, g in enumerate(G, 1):
        print(f"g{i} = {polynomial_to_string(g)}")
    
    quotients, remainder = divide_polynomial_by_set(S, G)
    
    print("\nQuotients:")
    for i, q in enumerate(quotients, 1):
        print(f"q{i} = {polynomial_to_string(q)}")
    
    print("\nRemainder:")
    print(f"r = {polynomial_to_string(remainder)}")
    
    print("\nVerification:")
    print(f"S = ", end="")
    terms = []
    for i, (q, g) in enumerate(zip(quotients, G), 1):
        terms.append(f"({polynomial_to_string(q)})({polynomial_to_string(g)})")
    terms.append(polynomial_to_string(remainder))
    print(" + ".join(terms))

def evaluate_polynomial(poly_dict, x_val, y_val):
    """
    Evaluate a polynomial at given x and y values.
    
    Parameters:
    poly_dict (dict): Polynomial represented as a dictionary of monomials
    x_val (float): Value of x
    y_val (float): Value of y
    
    Returns:
    float: Result of evaluation
    """
    result = 0
    for monomial, coeff in poly_dict.items():
        term = coeff * (x_val ** monomial[0]) * (y_val ** monomial[1])
        result += term
    return result

def plot_polynomials(S, G):
    """
    Create a 3D visualization of the polynomials.
    
    Parameters:
    S (dict): Input polynomial
    G (list of dict): List of divisor polynomials
    """
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Create meshgrid for x and y values
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # Plot input polynomial S
    ax1 = fig.add_subplot(131, projection='3d')
    Z_s = np.array([[evaluate_polynomial(S, xi, yi) for xi in x] for yi in y])
    surf = ax1.plot_surface(X, Y, Z_s, cmap='viridis')
    ax1.set_title(f'Input polynomial\n{polynomial_to_string(S)}')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Plot first polynomial in G
    ax2 = fig.add_subplot(132, projection='3d')
    Z_g1 = np.array([[evaluate_polynomial(G[0], xi, yi) for xi in x] for yi in y])
    surf = ax2.plot_surface(X, Y, Z_g1, cmap='viridis')
    ax2.set_title(f'g1 = {polynomial_to_string(G[0])}')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)
    
    # Plot second polynomial in G
    ax3 = fig.add_subplot(133, projection='3d')
    Z_g2 = np.array([[evaluate_polynomial(G[1], xi, yi) for xi in x] for yi in y])
    surf = ax3.plot_surface(X, Y, Z_g2, cmap='viridis')
    ax3.set_title(f'g2 = {polynomial_to_string(G[1])}')
    fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=5)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def s_polynomial(f, g):
    """
    Compute the S-polynomial of two polynomials f and g.
    
    Parameters:
    f, g (dict): Polynomials represented as dictionaries
    
    Returns:
    dict: S-polynomial
    """
    if not f or not g:
        return {}
        
    # Get leading terms
    lt_f = max(f.keys(), key=lambda k: (sum(k), k))
    lt_g = max(g.keys(), key=lambda k: (sum(k), k))
    
    # Compute LCM of leading terms
    lcm = lcm_monomials(lt_f, lt_g)
    
    # Compute multipliers
    mult_f = tuple(l - t for l, t in zip(lcm, lt_f))
    mult_g = tuple(l - t for l, t in zip(lcm, lt_g))
    
    # Compute coefficients
    coeff_f = f[lt_f]
    coeff_g = g[lt_g]
    
    # Initialize result
    result = {}
    
    # Multiply f by appropriate terms and add to result
    for m_f, c_f in f.items():
        new_monomial = tuple(m + mult for m, mult in zip(m_f, mult_f))
        result[new_monomial] = result.get(new_monomial, 0) + c_f * (coeff_g)
        
    # Subtract g multiplied by appropriate terms
    for m_g, c_g in g.items():
        new_monomial = tuple(m + mult for m, mult in zip(m_g, mult_g))
        result[new_monomial] = result.get(new_monomial, 0) - c_g * (coeff_f)
    
    # Remove zero terms
    return {k: v for k, v in result.items() if v != 0}

def buchberger_step(G):
    """
    Perform one step of Buchberger's algorithm.
    
    Parameters:
    G (list): Current Gröbner basis
    
    Returns:
    tuple: (new_polynomial, i, j) if a new polynomial is found, (None, -1, -1) otherwise
    """
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            s_poly = s_polynomial(G[i], G[j])
            if not s_poly:
                continue
                
            _, remainder = divide_polynomial_by_set(s_poly, G)
            
            if remainder:  # If remainder is not zero
                return remainder, i, j
                
    return None, -1, -1

def animate_buchberger(F):
    """
    Create an animated visualization of the Gröbner basis construction.
    
    Parameters:
    F (list): Initial set of polynomials
    """
    G = F.copy()
    all_steps = [G.copy()]
    step_descriptions = ["Initial polynomials"]
    
    # Collect all steps
    while True:
        new_poly, i, j = buchberger_step(G)
        if new_poly is None:
            break
        G.append(new_poly)
        all_steps.append(G.copy())
        step_descriptions.append(f"Added S-polynomial from g{i + 1} and g{j + 1}")
    
    # Find maximum number of polynomials
    max_polys = max(len(step) for step in all_steps)
    
    # Setup the figure
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, max_polys, height_ratios=[1, 4])
    
    # Text area for step description
    ax_text = plt.subplot(gs[0, :])
    ax_text.axis('off')
    
    # Create meshgrid for x and y values
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    def init():
        fig.clear()
        return []
    
    def animate(frame):
        fig.clear()
        gs = gridspec.GridSpec(2, max_polys, height_ratios=[1, 4])
        
        # Text area
        ax_text = plt.subplot(gs[0, :])
        ax_text.axis('off')
        ax_text.text(0.5, 0.5, f"Step {frame + 1}: {step_descriptions[frame]}", 
                    ha='center', va='center', fontsize=12)
        
        # Get current polynomials
        current_G = all_steps[frame]
        
        # Create subplots for each polynomial
        for i, g in enumerate(current_G):
            ax = fig.add_subplot(gs[1, i], projection='3d')
            Z = np.array([[evaluate_polynomial(g, xi, yi) for xi in x] for yi in y])
            surf = ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_title(f'g{i + 1} = {polynomial_to_string(g)}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        return []
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(all_steps), interval=2000,
                        blit=False, repeat=True)
    
    plt.show()
    return G

# Example usage
if __name__ == "__main__":
    # Define variables
    x, y = symbols('x y')

    # Initial polynomials
    F = [
        {(2, 0): 1, (1, 1): 2, (0, 2): 1},  # x^2 + 2xy + y^2
        {(1, 0): 1, (0, 1): 1},  # x + y
    ]

    # Animate the Gröbner basis construction
    G = animate_buchberger(F)
