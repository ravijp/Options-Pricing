#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Implements following methods in OptionsPricer Class:
        - The `blackscholes` method computes the fair option value based on the Black Schole pricing model
        - The `montecarlo` method runs several Monte Carlo simulations to converge on a fair option price

"""

__author__ = "Ravi Prakash"
__email__ = "ravijpp@gmail.com"


import numpy as np 
import scipy.stats as si 
import sympy as sy 
from random import gauss

np.seterr(divide = 'ignore') 

class OptionsPricer(object):
    def __init__(self, s, x, t, r, v, option_type):
        """
        Initialize an instance of the OptionsPricer class
        :param s: price of the underlying asset
        :param x: strike price of the option
        :param t: days to expiration
        :param r: risk free rate/rate of interest
        :param v: annualized volatility of the underlying
        :param option_type: "put" or "call"
        """
        self.s = s
        self.x = x
        self.t = t / 365        #keeping time to maturity in years
        self.r = r
        self.v = v
        self.option_type = option_type

    def _generate_asset_price(self):
        """ Calculate predicted Asset Price at the time of Option Expiry date.
        It used a random variable based on Gaus model and then calculate price using the below equation.
            St = S * exp((r− 0.5*σ^2)(T−t)+σT−t√ϵ)
        :return: <float> Expected Asset Price
        """
        expected_price = self.s * np.exp((self.r - 0.5 * self.v ** 2) * self.t + \
                        self.v * np.sqrt(self.t) * gauss(0.0, 1.0))
        return expected_price

    def _call_payoff(self, expected_price):
        """ Calculate payoff of the call option at Option Expiry Date assuming the asset price
        is equal to expected price. This calculation is based on below equation:
            Payoff at T = max(0,ExpectedPrice−Strike)
        :param expected_price: <float> Expected price of the underlying asset on Expiry Date
        :return: <float> payoff
        """
        return max(0, expected_price - self.x)

    def _put_payoff(self, expected_price):
        """ Calculate payoff of the put option at Option Expiry Date assuming the asset price
        is equal to expected price. This calculation is based on below equation:
            Payoff at T = max(0,Strike-ExpectedPrice)
        :param expected_price: <float> Expected price of the underlying asset on Expiry Date
        :return: <float> payoff
        """
        return max(0, self.x - expected_price)

    def _generate_simulations(self, iterations):
        """ Perform Brownian motion simulation to get the Call & Put option payouts on Expiry Date
        :return: <list of call-option payoffs>, <list of put-option payoffs>
        """
        call_payoffs, put_payoffs = [], []
        for _ in range(iterations):
            expected_asset_price = self._generate_asset_price()
            call_payoffs.append(self._call_payoff(expected_asset_price))
            put_payoffs.append(self._put_payoff(expected_asset_price))
        if self.option_type == "call":
            return call_payoffs
        elif self.option_type == "put":
            return put_payoffs

    def montecarlo(self, iterations):
        """
        User-facing option pricing method using monte-carlo simluations
        :return: price of the call/put option
        """
        payoffs = self._generate_simulations(iterations)
        discount_factor = np.exp(-1 * self.r * self.t)
        price = discount_factor * (sum(payoffs) / len(payoffs))
        return price

    def _calculate_d1(self):
        """ Famous d1 variable from Black-Scholes model calculated as shown in:
                https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
        :return: <float>
        """
        d1 = (np.log(self.s / self.x) +
              (self.r  + 0.5 * self.v ** 2) * self.t) / \
             (self.v * np.sqrt(self.t))
        return d1

    def _calculate_d2(self):
        """ Famous d2 variable from Black-Scholes model calculated as shown in:
                https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
        :return: <float>
        """
        d2 = (np.log(self.s / self.x) +
              (self.r - 0.5 * self.v ** 2) * self.t) / \
             (self.v * np.sqrt(self.t))
        return d2

    def blackscholes(self):
        """
        User-facing option pricing method using the Black Scholes algorithm
        :return: price of the call/put option
        """
        d1 = self._calculate_d1()
        d2 = self._calculate_d2()

        if self.option_type == "call": #price in case of call option
            result = (self.s * si.norm.cdf(d1, 0.0, 1.0) - self.x * \
                np.exp(-1 * self.r * self.t) * si.norm.cdf(d2, 0.0, 1.0))
        elif self.option_type == "put": #price in case of put option
            result = (self.x * np.exp(-1 * self.r * self.t) * \
                si.norm.cdf(-1 * d2, 0.0, 1.0) - self.s * si.norm.cdf(-1 * d1, 0.0, 1.0))
        else: #we raise exception if call or put are not provided as option type
            raise Exception('Option type should be selected either of "put" or "call".\
                 The value of Option type was: {}'.format(self.option_type))
        return result


if __name__ == '__main__':
    #   pricer = OptionsPricer(100, 110, 365, 0.02, 0.15, "put")
    pricer = OptionsPricer(200, 200, 365, 0.15, 0.1, 'call')
    print "MC = %.4f" % pricer.montecarlo(iterations=100000)
    print "BS = %.4f" % pricer.blackscholes()
