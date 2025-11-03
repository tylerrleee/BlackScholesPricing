from scipy.stats import norm
from abc import abstractmethod
import numpy as np

class BlackScholes_Base:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float):
        # Parameter checks
        assert S > 0.0, f"Asset price (S) needs to be larger than 0. Got '{S}'"
        assert K > 0.0, f"Strike price (K) needs to be larger than 0. Got '{K}'"
        assert T > 0.0, f"Time to maturity (T) needs to be larger than 0. Got '{T}'"
        assert sigma > 0.0, f"Volatility (sigma) needs to be larger than 0. Got '{sigma}'"
        assert q >= 0.0, f"Annual dividend yield (q) cannot be negative. Got '{q}'"

        # Model parameters
        self.S = S        # Asset price
        self.K = K        # Strike price
        self.T = T        # Time to maturity (in years)
        self.r = r        # Risk-free interest rate
        self.sigma = sigma # Volatility of the underlying asset
        self.q = q        # Annual dividend yield

    # Helper Functions 
    def _d1(self) -> float:
        """
        Computes the first intermediary term (d1) in the Black-Scholes model.

        Formula:
            d1 = [ln(S/K) + (r - q + 0.5 * sigma^2) * T] / (sigma * sqrt(T))

        Interpretation:
            - Represents the standardized distance (in standard deviations) between 
            the current log spot-to-strike ratio and the expected growth under 
            the risk-neutral measure.
            - Used as a weighting factor in the probability that the option 
            will be exercised (i.e., that the stock price exceeds the strike at maturity).
        """
        numerator = np.log(self.S / self.K) + (self.r - self.q + 0.50 * self.sigma**2) * self.T
        denominator = self.sigma * np.sqrt(self.T)
        return numerator / denominator
    
    def _d2(self) -> float:
        """
        Computes the second intermediary term (d2) in the Black-Scholes model.

        Formula:
            d2 = d1 - sigma * sqrt(T)

        Interpretation:
            - Represents the standardized log spot-to-strike ratio adjusted for volatility.
            - Corresponds to the risk-neutral probability that the option will expire 
            in-the-money, discounted at the risk-free rate.
        """
        return self._d1() - self.sigma * np.sqrt(self.T)


    @abstractmethod
    def price(self) -> float:
        """ Fair value for option"""
        ...

    @abstractmethod
    def in_the_money(self) -> float:
        """ Probability that option will be in the money at maturity"""
        ...
    
    @abstractmethod
    def forward_delta(self) -> float:
        """
        Def: Price sensitivity to the value of a forward contract
        Hedging Instruments: Forward or futures contracts
        
        """
        ...
    
    @abstractmethod
    def spot_delta(self) -> float:
        """
        Def: Option's price sensitivity to the spot price of the underlying
        Hedging Instruments: Phyiscal Underlying Asets (stocks, currency exchange)

        Arg: 
            Spot Price (float):
            Interest Rate Differentials:

        """
        ...

    
    def gamma(self) -> float:
        """
        Rate of change in Delta with respect to the underlying asset price (2nd derivative)
        """

        return (
            np.exp(-self.q * self.T) 
            * norm.cdf(self._d1) 
            / (self.S * self.sigma * np.sqrt(self.T))
        )