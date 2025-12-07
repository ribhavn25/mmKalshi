import abc
import time
from typing import Dict, List, Tuple
import logging
import uuid
import math
from pathlib import Path

import kalshi_python
from kalshi_python import KalshiClient, Configuration
from kalshi_python import exceptions as kalshi_exceptions

class AbstractTradingAPI(abc.ABC):
    @abc.abstractmethod
    def get_price(self) -> float:
        pass

    @abc.abstractmethod
    def place_order(self, action: str, side: str, price: float, quantity: int, expiration_ts: int = None) -> str:
        pass

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abc.abstractmethod
    def get_position(self) -> int:
        pass

    @abc.abstractmethod
    def get_orders(self) -> List[Dict]:
        pass

class KalshiTradingAPI(AbstractTradingAPI):
    def __init__(
        self,
        api_key_id: str,
        private_key_path: str,
        market_ticker: str,
        base_url: str,
        logger: logging.Logger,
    ):
        self.api_key_id = api_key_id
        self.private_key_path = private_key_path
        self.market_ticker = market_ticker
        self.logger = logger
        self.base_url = base_url or "https://api.elections.kalshi.com/trade-api/v2"
        self.client = self._create_client()

    def _create_client(self) -> KalshiClient:
        key_path = Path(self.private_key_path).expanduser()
        if not key_path.exists():
            raise FileNotFoundError(f"Private key not found at {key_path}")
        private_key_pem = key_path.read_text()

        configuration = Configuration(host=self.base_url)
        # kalshi_python.KalshiClient looks for these attributes on the configuration
        configuration.api_key_id = self.api_key_id
        configuration.private_key_pem = private_key_pem

        return kalshi_python.KalshiClient(configuration=configuration)

    def logout(self):
        # API key auth is stateless; nothing to tear down
        self.logger.info("Logout skipped (API key authentication)")

    def get_position(self) -> int:
        self.logger.info("Retrieving position...")
        response = self.client.get_positions(ticker=self.market_ticker)
        positions = response.positions or []
        net_position = 0
        for p in positions:
            self.logger.debug(f"Position detail: ticker={p.ticker}, position={p.position}, market_result={p.market_result}")
            if p.ticker != self.market_ticker:
                continue
            pos_val = int(p.position or 0)
            # If market_result is provided, treat 'no' as negative yes; otherwise just add the raw position
            if p.market_result == 'no':
                net_position -= pos_val
            else:
                net_position += pos_val
        self.logger.info(f"Current position (yes side): {net_position}")
        return net_position

    def get_price(self) -> Dict[str, float]:
        self.logger.info("Retrieving market data...")
        data = self.client.get_market(self.market_ticker)

        market = data.market
        if not market:
            raise ValueError(f"No market data returned for {self.market_ticker}")

        yes_bid = float(market.yes_bid or 0) / 100
        yes_ask = float(market.yes_ask or 0) / 100
        no_bid = float(market.no_bid or 0) / 100
        no_ask = float(market.no_ask or 0) / 100
        
        yes_mid_price = round((yes_bid + yes_ask) / 2, 2)
        no_mid_price = round((no_bid + no_ask) / 2, 2)

        self.logger.info(f"Current yes mid-market price: ${yes_mid_price:.2f}")
        self.logger.info(f"Current no mid-market price: ${no_mid_price:.2f}")
        return {"yes": yes_mid_price, "no": no_mid_price}

    def get_balance_cents(self) -> int:
        self.logger.info("Retrieving balance...")
        response = self.client.get_balance()
        balance_cents = int(response.balance or 0)
        self.logger.info(f"Current balance: ${balance_cents/100:.2f}")
        return balance_cents

    def place_order(self, action: str, side: str, price: float, quantity: int, expiration_ts: int = None) -> str:
        self.logger.info(f"Placing {action} order for {side} side at price ${price:.2f} with quantity {quantity}...")
        data = {
            "ticker": self.market_ticker,
            "action": action.lower(),  # 'buy' or 'sell'
            "type": "limit",
            "side": side,  # 'yes' or 'no'
            "count": quantity,
            "client_order_id": str(uuid.uuid4()),
        }
        price_to_send = int(price * 100) # Convert dollars to cents

        if side == "yes":
            data["yes_price"] = price_to_send
        else:
            data["no_price"] = price_to_send

        if expiration_ts is not None:
            data["expiration_ts"] = expiration_ts

        try:
            response = self.client.create_order(**data)
            order = response.order
            order_id = order.order_id if order else None
            self.logger.info(f"Placed {action} order for {side} side at price ${price:.2f} with quantity {quantity}, order ID: {order_id}")
            return str(order_id)
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            raise

    def cancel_order(self, order_id: int) -> bool:
        self.logger.info(f"Canceling order with ID {order_id}...")
        response = self.client.cancel_order(order_id=str(order_id))
        success = bool(response.reduced_by and response.reduced_by > 0)
        self.logger.info(f"Canceled order with ID {order_id}, success: {success}")
        return success

    def get_orders(self) -> List[Dict]:
        self.logger.info("Retrieving orders...")
        response = self.client.get_orders(ticker=self.market_ticker, status="resting")
        orders = [order.to_dict() for order in (response.orders or [])]
        self.logger.info(f"Retrieved {len(orders)} orders")
        return orders

class AvellanedaMarketMaker:
    def __init__(
        self,
        logger: logging.Logger,
        api: AbstractTradingAPI,
        gamma: float,
        k: float,
        sigma: float,
        T: float,
        max_position: int,
        order_expiration: int,
        min_spread: float = 0.01,
        position_limit_buffer: float = 0.1,
        inventory_skew_factor: float = 0.01,
        trade_side: str = "yes",
        max_order_size: int = None,
    ):
        self.api = api
        self.logger = logger
        self.base_gamma = gamma
        self.k = k
        self.sigma = sigma
        self.T = T
        self.max_position = max_position
        self.order_expiration = order_expiration
        self.min_spread = min_spread
        self.position_limit_buffer = position_limit_buffer
        self.inventory_skew_factor = inventory_skew_factor
        self.trade_side = trade_side
        self.max_order_size = max_order_size

    def run(self, dt: float):
        start_time = time.time()
        while time.time() - start_time < self.T:
            current_time = time.time() - start_time
            self.logger.info(f"Running Avellaneda market maker at {current_time:.2f}")

            mid_prices = self.api.get_price()
            mid_price = mid_prices[self.trade_side]
            inventory = self.api.get_position()
            self.logger.info(f"Current mid price for {self.trade_side}: {mid_price:.4f}, Inventory: {inventory}")

            reservation_price = self.calculate_reservation_price(mid_price, inventory, current_time)
            bid_price, ask_price = self.calculate_asymmetric_quotes(mid_price, inventory, current_time)
            buy_size, sell_size = self.calculate_order_sizes(inventory)

            self.logger.info(f"Reservation price: {reservation_price:.4f}")
            self.logger.info(f"Computed desired bid: {bid_price:.4f}, ask: {ask_price:.4f}")

            self.manage_orders(bid_price, ask_price, buy_size, sell_size, inventory)

            time.sleep(dt)

        self.logger.info("Avellaneda market maker finished running")

    def calculate_asymmetric_quotes(self, mid_price: float, inventory: int, t: float) -> Tuple[float, float]:
        reservation_price = self.calculate_reservation_price(mid_price, inventory, t)
        base_spread = self.calculate_optimal_spread(t, inventory)
        
        position_ratio = inventory / self.max_position
        spread_adjustment = base_spread * abs(position_ratio) * 3
        
        if inventory > 0:
            bid_spread = base_spread / 2 + spread_adjustment
            ask_spread = max(base_spread / 2 - spread_adjustment, self.min_spread / 2)
        else:
            bid_spread = max(base_spread / 2 - spread_adjustment, self.min_spread / 2)
            ask_spread = base_spread / 2 + spread_adjustment
        
        bid_price = max(0, min(mid_price, reservation_price - bid_spread))
        ask_price = min(1, max(mid_price, reservation_price + ask_spread))
        
        return bid_price, ask_price

    def calculate_reservation_price(self, mid_price: float, inventory: int, t: float) -> float:
        dynamic_gamma = self.calculate_dynamic_gamma(inventory)
        inventory_skew = inventory * self.inventory_skew_factor * mid_price
        return mid_price + inventory_skew - inventory * dynamic_gamma * (self.sigma**2) * (1 - t/self.T)

    def calculate_optimal_spread(self, t: float, inventory: int) -> float:
        dynamic_gamma = self.calculate_dynamic_gamma(inventory)
        base_spread = (dynamic_gamma * (self.sigma**2) * (1 - t/self.T) + 
                       (2 / dynamic_gamma) * math.log(1 + (dynamic_gamma / self.k)))
        position_ratio = abs(inventory) / self.max_position
        spread_adjustment = 1 - (position_ratio ** 2)
        return max(base_spread * spread_adjustment * 0.01, self.min_spread)

    def calculate_dynamic_gamma(self, inventory: int) -> float:
        position_ratio = inventory / self.max_position
        return self.base_gamma * math.exp(-abs(position_ratio))

    def calculate_order_sizes(self, inventory: int) -> Tuple[int, int]:
        remaining_capacity = max(self.max_position - max(inventory, 0), 0)
        buffer_size = int(self.max_position * self.position_limit_buffer)

        # Buys are limited by remaining capacity and max_order_size
        buy_size = max(0, min(self.max_position, remaining_capacity))
        if self.max_order_size:
            buy_size = min(buy_size, self.max_order_size)
        if buy_size == 0 and remaining_capacity > 0:
            buy_size = 1  # place a minimal buy if we have any capacity

        # Sells are limited by owned inventory and max_order_size
        if inventory > 0:
            sell_size = max(0, min(inventory, buffer_size if buffer_size > 0 else inventory))
            if self.max_order_size:
                sell_size = min(sell_size, self.max_order_size)
        else:
            sell_size = 0
        
        return buy_size, sell_size

    def manage_orders(self, bid_price: float, ask_price: float, buy_size: int, sell_size: int, inventory: int):
        current_orders = self.api.get_orders()
        self.logger.info(f"Retrieved {len(current_orders)} total orders")

        buy_orders = []
        sell_orders = []
        opposite_orders = []

        for order in current_orders:
            if order['side'] == self.trade_side:
                if order['action'] == 'buy':
                    buy_orders.append(order)
                elif order['action'] == 'sell':
                    sell_orders.append(order)
            else:
                opposite_orders.append(order)

        # Cancel any orders on the opposite side to ensure we only trade one side
        for order in opposite_orders:
            try:
                self.logger.info(f"Cancelling opposite-side order {order['order_id']} side {order['side']}")
                self.api.cancel_order(order['order_id'])
            except kalshi_exceptions.NotFoundException:
                self.logger.info(f"Opposite-side order {order['order_id']} already gone when canceling")
            except Exception as e:
                self.logger.error(f"Failed to cancel opposite-side order {order['order_id']}: {e}")

        self.logger.info(f"Current buy orders: {len(buy_orders)}")
        self.logger.info(f"Current sell orders: {len(sell_orders)}")
        self.logger.info(f"Desired buy size: {buy_size}, Desired sell size: {sell_size}, Side inventory: {inventory}")

        # Handle buy orders
        if buy_size >= 1:
            self.handle_order_side('buy', buy_orders, bid_price, buy_size, inventory)
        else:
            self.logger.info("Skipping buy management; desired buy size < 1")

        # Handle sell orders
        if sell_size >= 1:
            self.handle_order_side('sell', sell_orders, ask_price, sell_size, inventory)
        else:
            self.logger.info("Skipping sell management; desired sell size < 1")

    def handle_order_side(self, action: str, orders: List[Dict], desired_price: float, desired_size: int, inventory: int):
        keep_order = None
        for order in orders:
            current_price = float(order['yes_price']) / 100 if self.trade_side == 'yes' else float(order['no_price']) / 100
            if keep_order is None and abs(current_price - desired_price) < 0.01 and order['remaining_count'] == desired_size:
                keep_order = order
                self.logger.info(f"Keeping existing {action} order. ID: {order['order_id']}, Price: {current_price:.4f}")
            else:
                self.logger.info(f"Cancelling extraneous {action} order. ID: {order['order_id']}, Price: {current_price:.4f}")
                try:
                    self.api.cancel_order(order['order_id'])
                except kalshi_exceptions.NotFoundException:
                    # Order already gone; not fatal.
                    self.logger.info(f"{action.capitalize()} order {order['order_id']} already gone when canceling")
                except Exception as e:
                    self.logger.error(f"Failed to cancel {action} order {order['order_id']}: {e}")
                    return

        current_price = self.api.get_price()[self.trade_side]
        if keep_order is None:
            if (action == 'buy' and desired_price < current_price) or (action == 'sell' and desired_price > current_price):
                if action == 'buy':
                    available_cents = self.api.get_balance_cents()
                    required_cents = int(desired_price * 100 * desired_size)
                    if required_cents > available_cents:
                        self.logger.info(f"Skipping buy order; needed ${required_cents/100:.2f} > available ${available_cents/100:.2f}")
                        return
                else:
                    available_inventory = max(inventory, 0)
                    if desired_size > available_inventory:
                        self.logger.info(f"Skipping sell order; desired size {desired_size} > available inventory {available_inventory}")
                        return
                try:
                    order_id = self.api.place_order(action, self.trade_side, desired_price, desired_size, int(time.time()) + self.order_expiration)
                    self.logger.info(f"Placed new {action} order. ID: {order_id}, Price: {desired_price:.4f}, Size: {desired_size}")
                except Exception as e:
                    self.logger.error(f"Failed to place {action} order: {str(e)}")
            else:
                self.logger.info(f"Skipped placing {action} order. Desired price {desired_price:.4f} does not improve on current price {current_price:.4f}")
