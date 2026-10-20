"""Hardware lots with delivery dates, age-dependent value and finite useful life.

All costs and hash units are scenario units, not a live ASIC price calibration.
"""
from dataclasses import dataclass


@dataclass
class HardwareLot:
    working: float
    damaged: float
    unit_cost: float
    age_days: int = 0


@dataclass
class HardwareOrder:
    hash_power: float
    cost: float
    delivery_day: int


class HardwareFleet:
    def __init__(self, initial_hash, unit_cost, params):
        self.params = params
        self.lots = [HardwareLot(initial_hash, 0.0, unit_cost, params.initial_hardware_age_days)]
        self.orders = []

    @property
    def working(self):
        return sum(lot.working for lot in self.lots)

    @property
    def damaged(self):
        return sum(lot.damaged for lot in self.lots)

    @property
    def pending_hash(self):
        return sum(order.hash_power for order in self.orders)

    def unit_resale(self, lot):
        fraction = min(1.0, lot.age_days / self.params.hardware_lifetime_days)
        factor = self.params.new_hardware_resale_fraction + fraction * (
            self.params.residual_value_fraction - self.params.new_hardware_resale_fraction
        )
        return lot.unit_cost * factor

    def value(self):
        p = self.params
        installed = sum(
            lot.working * self.unit_resale(lot)
            + lot.damaged * max(0.0, self.unit_resale(lot) - p.repair_cost_per_hash)
            for lot in self.lots
        )
        # Orders are prepaid assets, not productive equipment, until delivery.
        return installed + sum(order.cost for order in self.orders)

    def advance_day(self, day):
        """Age installed lots, retire expired capacity and commission due orders."""
        retired_hash = 0.0
        salvage_cash = 0.0
        surviving = []
        for lot in self.lots:
            lot.age_days += 1
            if lot.age_days >= self.params.hardware_lifetime_days:
                retired_hash += lot.working + lot.damaged
                salvage_cash += lot.working * self.unit_resale(lot)
                salvage_cash += lot.damaged * max(0, self.unit_resale(lot) - self.params.repair_cost_per_hash)
            else:
                surviving.append(lot)
        self.lots = surviving
        delivered_hash = 0.0
        remaining = []
        for order in self.orders:
            if order.delivery_day <= day:
                self.lots.append(HardwareLot(order.hash_power, 0.0, order.cost / order.hash_power))
                delivered_hash += order.hash_power
            else:
                remaining.append(order)
        self.orders = remaining
        return delivered_hash, retired_hash, salvage_cash

    def order(self, hash_power, cost, day):
        if hash_power <= 0 or cost <= 0:
            raise ValueError("Orders must have positive capacity and cost")
        self.orders.append(HardwareOrder(hash_power, cost, day + self.params.delivery_delay_days))

    def damage(self, fraction):
        lost = 0.0
        for lot in self.lots:
            amount = lot.working * fraction
            lot.working -= amount
            lot.damaged += amount
            lost += amount
        return lost

    def repair(self, budget):
        """Repair oldest lots first within the available cash budget."""
        repaired = 0.0
        cost = 0.0
        for lot in sorted(self.lots, key=lambda item: item.age_days, reverse=True):
            amount = min(lot.damaged, max(0, budget - cost) / self.params.repair_cost_per_hash)
            lot.damaged -= amount
            lot.working += amount
            repaired += amount
            cost += amount * self.params.repair_cost_per_hash
        return repaired, cost

    def sell(self, amount):
        """Dispose of oldest working capacity at its current age-dependent value."""
        sold = 0.0
        cash = 0.0
        for lot in sorted(self.lots, key=lambda item: item.age_days, reverse=True):
            quantity = min(lot.working, max(0, amount - sold))
            lot.working -= quantity
            sold += quantity
            cash += quantity * self.unit_resale(lot)
        self.lots = [lot for lot in self.lots if lot.working + lot.damaged > 1e-10]
        return sold, cash

    def liquidate(self):
        """Exit sells installed assets and cancels orders with a stated haircut."""
        installed_value = self.value() - sum(order.cost for order in self.orders)
        refunds = sum(order.cost for order in self.orders) * self.params.order_cancellation_refund
        self.lots = []
        self.orders = []
        return installed_value + refunds
