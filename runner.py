import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict

import yaml
from dotenv import load_dotenv

from mm import AvellanedaMarketMaker, KalshiTradingAPI


def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def create_api(api_config, logger):
    api_key_id = os.getenv("KALSHI_API_KEY_ID")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    base_url = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")

    missing = [name for name, value in {
        "KALSHI_API_KEY_ID": api_key_id,
        "KALSHI_PRIVATE_KEY_PATH": private_key_path,
        "KALSHI_BASE_URL": base_url,
    }.items() if not value]

    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    return KalshiTradingAPI(
        api_key_id=api_key_id,
        private_key_path=private_key_path,
        market_ticker=api_config['market_ticker'],
        base_url=base_url,
        logger=logger,
    )


def create_market_maker(mm_config, api, logger, trade_side: str):
    if trade_side not in ("yes", "no"):
        raise ValueError(f"trade_side must be 'yes' or 'no', got {trade_side}")
    return AvellanedaMarketMaker(
        logger=logger,
        api=api,
        gamma=mm_config.get('gamma', 0.1),
        k=mm_config.get('k', 1.5),
        sigma=mm_config.get('sigma', 0.5),
        T=mm_config.get('T', 3600),
        max_position=mm_config.get('max_position', 100),
        order_expiration=mm_config.get('order_expiration', 300),
        min_spread=mm_config.get('min_spread', 0.01),
        position_limit_buffer=mm_config.get('position_limit_buffer', 0.1),
        inventory_skew_factor=mm_config.get('inventory_skew_factor', 0.01),
        trade_side=trade_side,
        max_order_size=mm_config.get('max_order_size')
    )

def run_strategy(config_name: str, config: Dict):
    # Create a logger for this specific strategy
    logger = logging.getLogger(f"Strategy_{config_name}")
    logger.setLevel(config.get('log_level', 'INFO'))

    # Create file handler
    fh = logging.FileHandler(f"{config_name}.log")
    fh.setLevel(config.get('log_level', 'INFO'))
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(config.get('log_level', 'INFO'))
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Starting strategy: {config_name}")

    api = None
    try:
        # Create API
        api = create_api(config['api'], logger)

        # Create market maker
        market_maker = create_market_maker(
            config['market_maker'],
            api,
            logger,
            trade_side=config['api'].get('trade_side', 'yes')
        )

        # Run market maker
        market_maker.run(config.get('dt', 1.0))
    except KeyboardInterrupt:
        logger.info("Market maker stopped by user")
    except Exception as e:
        logger.exception(f"An error occurred in strategy {config_name}: {str(e)}")
    finally:
        # Ensure logout happens even if an exception occurs
        if api:
            try:
                api.logout()
            except Exception:
                logger.exception("Failed to logout cleanly")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi Market Making Algorithm")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load all configurations
    configs = load_config(args.config)

    # Load environment variables
    load_dotenv()

    # Print the name of every strategy being run
    print("Starting the following strategies:")
    for config_name in configs:
        print(f"- {config_name}")

    # Run all strategies in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(configs)) as executor:
        for config_name, config in configs.items():
            executor.submit(run_strategy, config_name, config)
