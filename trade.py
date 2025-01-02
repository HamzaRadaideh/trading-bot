import os
import math
import logging
import schedule
import time
from datetime import datetime, date
import pytz
from typing import Optional, Tuple, Dict

import alpaca_trade_api as tradeapi

# --------------------------------------------------------------------------
#  CONFIGURATION
# --------------------------------------------------------------------------

# You could load these from environment variables or a config file
API_KEY = os.getenv("APCA_API_KEY_ID", "YOUR_API_KEY")
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "YOUR_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading endpoint

SYMBOL = "AAPL"
BUY_THRESHOLD = -0.01          # e.g. Buy if price is down 1% from previous close
SELL_THRESHOLD = 0.02          # e.g. Sell if up 2% from purchase price
INVESTMENT_PERCENTAGE = 0.5    # 50% of available buying power
STOP_LOSS_PERCENTAGE = 0.02    # 2% below the entry price
TAKE_PROFIT_PERCENTAGE = 0.03  # 3% above the entry price
CHECK_INTERVAL_SECONDS = 60    # How frequently to run the trading loop (in seconds)

# --------------------------------------------------------------------------
#  LOGGING SETUP
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBot")


# --------------------------------------------------------------------------
#  TRADING BOT CLASS
# --------------------------------------------------------------------------
class TradingBot:
    """
    A trading bot using Alpaca's API with bracket orders for stop-loss and take-profit.
    """

    def __init__(
        self,
        symbol: str,
        buy_threshold: float,
        sell_threshold: float,
        investment_percentage: float,
        stop_loss_percentage: float,
        take_profit_percentage: float,
        api_key: str,
        api_secret: str,
        base_url: str
    ):
        """
        Initialize the trading bot with configuration parameters.
        
        Args:
            symbol: Stock symbol to trade
            buy_threshold: Percentage drop from previous close to trigger buy
            sell_threshold: Percentage gain (unrealized) at which to consider selling
            investment_percentage: Fraction of available buying power to invest
            stop_loss_percentage: Fraction below entry price for stop-loss
            take_profit_percentage: Fraction above entry price for take-profit
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            base_url: Alpaca API base URL (paper or live trading)
        """
        self.symbol = symbol
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.investment_percentage = investment_percentage
        self.stop_loss_percentage = stop_loss_percentage
        self.take_profit_percentage = take_profit_percentage

        self.api = tradeapi.REST(
            key_id=api_key,
            secret_key=api_secret,
            base_url=base_url,
            api_version='v2'
        )

        # Keep track of whether we have a position for the symbol
        self.has_position = False
        self.position_qty = 0
        self.position_avg_entry_price = 0.0
        self._market_hours_date = None
        self._market_open_time = None
        self._market_close_time = None

    # ----------------------------------------------------------------------
    #  HELPER FUNCTIONS
    # ----------------------------------------------------------------------
    def get_market_hours(self) -> Tuple[datetime, datetime]:
        """
        Get the market open/close time for the current date.
        Caches results to avoid repeated API calls.
        """
        today = date.today()
        ny_tz = pytz.timezone('America/New_York')

        if self._market_hours_date == today and self._market_open_time and self._market_close_time:
            return self._market_open_time, self._market_close_time

        # Fetch today's calendar
        calendar = self.api.get_calendar(start=str(today), end=str(today))
        if not calendar:
            # If no market data, assume closed or holiday
            now_dt = datetime.now(ny_tz)
            return now_dt, now_dt

        market_open_str = calendar[0].open
        market_close_str = calendar[0].close

        # Convert strings to aware datetime objects
        # Alpaca times are in local NY time by default
        market_open_dt = ny_tz.localize(datetime.strptime(market_open_str, "%Y-%m-%dT%H:%M:%S-%H:%M"))
        market_close_dt = ny_tz.localize(datetime.strptime(market_close_str, "%Y-%m-%dT%H:%M:%S-%H:%M"))

        self._market_hours_date = today
        self._market_open_time = market_open_dt
        self._market_close_time = market_close_dt

        return market_open_dt, market_close_dt

    def is_market_open(self) -> bool:
        """
        Check if the market is open right now.
        """
        now = datetime.now(pytz.timezone('America/New_York'))
        open_time, close_time = self.get_market_hours()
        return open_time <= now <= close_time

    def get_buying_power(self) -> float:
        """
        Returns the account's current buying power.
        """
        try:
            account = self.api.get_account()
            return float(account.buying_power)
        except Exception as e:
            logger.error(f"[get_buying_power] Error: {e}")
            return 0.0

    def get_current_price(self) -> Optional[float]:
        """
        Returns the current (latest) trade price for self.symbol.
        """
        try:
            latest_trade = self.api.get_latest_trade(self.symbol)
            return latest_trade.price
        except Exception as e:
            logger.error(f"[get_current_price] Error: {e}")
            return None

    def get_previous_close_price(self) -> Optional[float]:
        """
        Get the previous close price for self.symbol using daily bars.
        """
        try:
            barset = self.api.get_bars(self.symbol, '1Day', limit=2)
            # Usually the last bar in the list is today's. The one before is the previous close.
            # But we have to confirm the time is in the past to get the correct bar.
            if len(barset) < 2:
                return None
            # Typically barset[-2] is yesterday, barset[-1] is today
            previous_bar = barset[-2]
            return previous_bar.c
        except Exception as e:
            logger.error(f"[get_previous_close_price] Error: {e}")
            return None

    def update_position_status(self):
        """
        Update self.has_position, self.position_qty, self.position_avg_entry_price
        by querying Alpaca for the current open position in self.symbol.
        """
        try:
            position = self.api.get_position(self.symbol)
            self.has_position = True
            self.position_qty = float(position.qty)
            self.position_avg_entry_price = float(position.avg_entry_price)
        except tradeapi.rest.APIError:
            # Means no open position
            self.has_position = False
            self.position_qty = 0
            self.position_avg_entry_price = 0.0
        except Exception as e:
            logger.error(f"[update_position_status] Unexpected error: {e}")

    def calculate_unrealized_plpc(self) -> float:
        """
        Calculate the percentage gain/loss on the current position.
        Returns 0.0 if no position.
        """
        if not self.has_position or self.position_qty == 0:
            return 0.0
        current_price = self.get_current_price()
        if not current_price:
            return 0.0
        # Percentage from avg entry price
        return (current_price - self.position_avg_entry_price) / self.position_avg_entry_price

    # ----------------------------------------------------------------------
    #  ORDER FUNCTIONS
    # ----------------------------------------------------------------------
    def place_bracket_order(self, qty: int, limit_price: float):
        """
        Place a bracket order with attached stop-loss and take-profit.

        Args:
            qty (int): quantity of shares to buy
            limit_price (float): limit price for the buy order

        The bracket order includes:
            - Main buy limit order
            - Stop-loss at (1 - stop_loss_percentage) * limit_price
            - Take-profit at (1 + take_profit_percentage) * limit_price
        """
        # Construct bracket order
        stop_loss_price = round(limit_price * (1 - self.stop_loss_percentage), 2)
        take_profit_price = round(limit_price * (1 + self.take_profit_percentage), 2)

        order_data = {
            "symbol": self.symbol,
            "qty": qty,
            "side": "buy",
            "type": "limit",
            "time_in_force": "day",
            "limit_price": round(limit_price, 2),
            "order_class": "bracket",
            "take_profit": {
                "limit_price": take_profit_price
            },
            "stop_loss": {
                "stop_price": stop_loss_price
            }
        }

        try:
            order = self.api.submit_order(**order_data)
            logger.info(
                f"[place_bracket_order] Placed bracket BUY order for {qty} shares of {self.symbol} "
                f"at limit={limit_price:.2f}, TP={take_profit_price:.2f}, SL={stop_loss_price:.2f}."
            )
            return order
        except Exception as e:
            logger.error(f"[place_bracket_order] Error placing bracket order: {e}")
            return None

    def place_sell_order(self, qty: float):
        """
        Place a sell order (could be a market or limit). Here we do a simple market sell.
        """
        try:
            order = self.api.submit_order(
                symbol=self.symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
            logger.info(f"[place_sell_order] Placed MARKET SELL order for {qty} shares of {self.symbol}.")
            return order
        except Exception as e:
            logger.error(f"[place_sell_order] Error: {e}")
            return None

    # ----------------------------------------------------------------------
    #  MAIN TRADING LOGIC
    # ----------------------------------------------------------------------
    def run_trading_cycle(self):
        """
        Executes one cycle of the trading strategy:
          1. Check if market is open.
          2. Get current price & previous close price.
          3. Decide whether to buy or sell based on thresholds.
          4. Update position status.
        """
        if not self.is_market_open():
            logger.info("[run_trading_cycle] Market is closed. Skipping this cycle.")
            return

        current_price = self.get_current_price()
        previous_close_price = self.get_previous_close_price()

        if not current_price or not previous_close_price:
            logger.warning("[run_trading_cycle] Unable to get valid prices. Skipping iteration.")
            return

        # Update position status from broker
        self.update_position_status()

        # If no position, check if we need to buy
        if not self.has_position:
            # Compare the current price movement vs previous close
            percentage_change = (current_price - previous_close_price) / previous_close_price
            logger.info(
                f"[run_trading_cycle] Current price={current_price:.2f}, "
                f"Prev close={previous_close_price:.2f}, "
                f"Change={percentage_change * 100:.2f}%"
            )

            if percentage_change <= self.buy_threshold:
                # Calculate how many shares to buy
                buying_power = self.get_buying_power()
                investment_amount = buying_power * self.investment_percentage
                qty_to_buy = math.floor(investment_amount / current_price)

                if qty_to_buy <= 0:
                    logger.info(
                        "[run_trading_cycle] No sufficient buying power to purchase shares. "
                        f"Needed > 0, got={qty_to_buy}."
                    )
                    return

                # Place bracket order (buy limit) just slightly above the current price
                limit_price = current_price * 1.001  # e.g. 0.1% above current to improve fill chance
                self.place_bracket_order(qty_to_buy, limit_price)
            else:
                logger.info("[run_trading_cycle] Buy conditions not met.")
        
        else:
            # If we already have a position, check if we want to manually exit early
            # if the unrealized gain exceeds some threshold. 
            # However, bracket orders typically will auto-sell at the stop-loss or take-profit.
            # This logic is here if you want to override or close the position early.
            unrealized_plpc = self.calculate_unrealized_plpc()  # e.g. 0.02 for 2% gain
            logger.info(
                f"[run_trading_cycle] Already have position. Unrealized P/L% = {unrealized_plpc*100:.2f}%"
            )

            if unrealized_plpc >= self.sell_threshold:
                logger.info(
                    f"[run_trading_cycle] Manual SELL threshold triggered at {unrealized_plpc*100:.2f}% gain. "
                    "Placing sell order..."
                )
                self.place_sell_order(self.position_qty)
            else:
                logger.info("[run_trading_cycle] Holding position; no manual exit triggered.")

    def run_forever(self):
        """
        Schedule the trading cycle to run periodically.
        """
        logger.info(f"[run_forever] Starting trading bot for {self.symbol}...")

        # Use schedule to run our trading cycle every X seconds
        schedule.every(CHECK_INTERVAL_SECONDS).seconds.do(self.run_trading_cycle)

        while True:
            schedule.run_pending()
            time.sleep(1)


# --------------------------------------------------------------------------
#  ENTRY POINT
# --------------------------------------------------------------------------
if __name__ == "__main__":
    bot = TradingBot(
        symbol=SYMBOL,
        buy_threshold=BUY_THRESHOLD,
        sell_threshold=SELL_THRESHOLD,
        investment_percentage=INVESTMENT_PERCENTAGE,
        stop_loss_percentage=STOP_LOSS_PERCENTAGE,
        take_profit_percentage=TAKE_PROFIT_PERCENTAGE,
        api_key=API_KEY,
        api_secret=API_SECRET,
        base_url=BASE_URL
    )

    # Start the main loop (scheduled tasks)
    bot.run_forever()
