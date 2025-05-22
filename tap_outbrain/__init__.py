#!/usr/bin/env python3

from decimal import Decimal
from typing import Dict, List, Any, Optional

import argparse
import base64
import copy
import datetime
import dateutil.parser
import json
import os
import sys
import time

import backoff
import requests
import singer

import tap_outbrain.schemas as schemas

logger = singer.get_logger()

BASE_URL = 'https://api.outbrain.com/amplify/v0.1'

DEFAULT_STATE = {
    'campaign_performance_outbrain': {},
    'link_performance_outbrain': {}
}

DEFAULT_START_DATE = '2020-01-01'

def giveup(error):
    logger.error(error.response.text)
    response = error.response
    return not (response.status_code == 429 or
                response.status_code >= 500)

@backoff.on_exception(backoff.constant,
                      (requests.exceptions.RequestException),
                      jitter=backoff.random_jitter,
                      max_tries=5,
                      giveup=giveup,
                      interval=30)
def request(url, access_token, params={}):
    logger.info("Making request: GET {} {}".format(url, params))

    try:
        response = requests.get(
            url,
            headers={'OB-TOKEN-V1': access_token},
            params=params)
    except Exception as e:
        logger.exception(e)
        raise

    logger.info("Got response code: {}".format(response.status_code))

    response.raise_for_status()
    return response


def generate_token(username, password):
    logger.info("Generating new token using basic auth.")

    credentials = f"{username}:{password}"
    encoded = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')

    response = requests.get(
        '{}/login'.format(BASE_URL),
        headers={'Authorization': 'Basic {}'.format(encoded)})
    response.raise_for_status()

    logger.info("Got response code: {}".format(response.status_code))

    return response.json().get('OB-TOKEN-V1')


def parse_datetime(datetime_str):
    """Parse datetime string to ISO format with None handling."""
    if datetime_str is None:
        return None
    try:
        dt = dateutil.parser.parse(datetime_str)
        return dt.isoformat('T') + 'Z'
    except (ValueError, TypeError):
        return None

def decimal_to_float(obj):
    """Recursively convert Decimals to floats."""
    if isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_float(x) for x in obj]
    elif isinstance(obj, Decimal):
        return float(obj)
    return obj


def get_all_marketers(access_token):
    logger.info("Fetching list of all accessible marketers...")
    try:
        response = request(
            f'{BASE_URL}/marketers',
            access_token
        )
        marketers_data = response.json().get('marketers', [])
        if not marketers_data:
            logger.warning("No marketers found for the provided access token.")
            return []
        
        extracted_marketers = []
        for m_info in marketers_data:
            m_id = m_info.get('id')
            m_name = m_info.get('name')
            if m_id and m_name:
                extracted_marketers.append({'id': m_id, 'name': m_name})
            else:
                logger.warning(f"Marketer data missing id or name: {m_info}")
        
        logger.info(f"Found {len(extracted_marketers)} marketers.")
        return extracted_marketers
    except Exception as e:
        logger.error(f"Error fetching marketers: {str(e)}")
        logger.debug(e, exc_info=True)
        return []


def parse_performance(result: Dict[str, Any], extra_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Parse performance metrics with decimal handling."""
    # Convert nested decimals in both result and extra_fields
    converted_result = decimal_to_float(result)
    converted_extra = decimal_to_float(extra_fields)
    
    metrics = converted_result.get('metrics', {})
    metadata = converted_result.get('metadata', {})

    return {
        'fromDate': metadata.get('fromDate'),
        'impressions': int(metrics.get('impressions', 0)),
        'clicks': int(metrics.get('clicks', 0)),
        'ctr': float(metrics.get('ctr', 0.0)),
        'spend': float(metrics.get('spend', 0.0)),
        'ecpc': float(metrics.get('ecpc', 0.0)),
        'conversions': int(metrics.get('conversions', 0)),
        'conversionRate': float(metrics.get('conversionRate', 0.0)),
        'cpa': float(metrics.get('cpa', 0.0)),
        'account_id': extra_fields.get('account_id', ''),
        'account_name': extra_fields.get('account_name', ''),
        # Add converted extra fields
        **converted_extra
    }

def get_date_ranges(start, end, interval_in_days):
    if start > end:
        return []

    to_return = []
    interval_start = start

    while interval_start < end:
        to_return.append({
            'from_date': interval_start,
            'to_date': min(end,
                           (interval_start + datetime.timedelta(
                               days=interval_in_days-1)))
        })

        interval_start = interval_start + datetime.timedelta(
            days=interval_in_days)

    return to_return

def sync_campaign_performance(state, access_token, current_marketer_id, current_marketer_name, config):
    """
    Fetch aggregated campaign performance for all campaigns of a specific marketer over the configured date range.
    """
    logger.info(f"Starting campaign performance sync for marketer ID: {current_marketer_id} (Name: {current_marketer_name})")

    from_date = config['start_date']
    # Use end_date from config if provided, otherwise use today's date
    to_date = config.get('end_date') or datetime.date.today().isoformat()
    
    params = {
        'from': from_date,
        'to': to_date,
        'limit': 500
    }

    logger.info(f"Fetching campaign reports for marketer {current_marketer_id} ({current_marketer_name}) from {from_date} to {to_date} with limit {params['limit']}")
    
    api_url = f"{BASE_URL}/reports/marketers/{current_marketer_id}/campaigns"
    response = request(
        api_url,
        access_token,
        params
    )
    results = response.json().get('results', [])
    
    logger.info(f"Received {len(results)} campaign results from API for marketer {current_marketer_id}.")

    records = []
    for c_result in results:
        metadata = c_result.get('metadata', {})
        metrics = c_result.get('metrics', {})

        def get_metric(metric_name, default_val_if_none=None, cast_to_type=float):
            val = metrics.get(metric_name)
            if val is None:
                return default_val_if_none
            try:
                if cast_to_type == int:
                    return int(float(val)) 
                return cast_to_type(val)
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not cast metric '{metric_name}' with value {repr(val)} "
                    f"for campaign {metadata.get('id')} to {cast_to_type}. Using default: {default_val_if_none}."
                )
                return default_val_if_none
        
        record = {
            'marketer_id': current_marketer_id,
            'marketer_name': current_marketer_name,
            'campaign_id': metadata.get('id'),
            'campaign_name': metadata.get('name'),
            'impressions': get_metric('impressions', 0, int),
            'clicks': get_metric('clicks', 0, int),
            'totalConversions': get_metric('totalConversions', 0, int),
            'conversions': get_metric('conversions', 0, int),
            'viewConversions': get_metric('viewConversions', 0, int),
            'spend': get_metric('spend', 0.0, float),
            'ecpc': get_metric('ecpc', 0.0, float),
            'ctr': get_metric('ctr', 0.0, float),
            'dstFeeCost': get_metric('dstFeeCost', 0.0, float),
            'conversionRate': get_metric('conversionRate', 0.0, float),
            'viewConversionRate': get_metric('viewConversionRate', 0.0, float),
            'cpa': get_metric('cpa', 0.0, float),
            'totalCpa': get_metric('totalCpa', 0.0, float),
            'totalSumValue': get_metric('totalSumValue', 0.0, float),
            'sumValue': get_metric('sumValue', 0.0, float),
            'viewSumValue': get_metric('viewSumValue', 0.0, float),
            'totalAverageValue': get_metric('totalAverageValue', 0.0, float),
            'averageValue': get_metric('averageValue', 0.0, float),
            'viewAverageValue': get_metric('viewAverageValue', 0.0, float),
            'totalRoas': get_metric('totalRoas', 0.0, float),
            'roas': get_metric('roas', 0.0, float),
            'optimizedConversions': get_metric('optimizedConversions', 0, int),
            'fetched_from': from_date,
            'fetched_to': to_date
        }
        records.append(record)

    if records:
        singer.write_records('campaign_performance_outbrain', records)
        logger.info(f"Wrote {len(records)} records to 'campaign_performance_outbrain' stream for marketer {current_marketer_id}.")
    else:
        logger.info(f"No records to write for 'campaign_performance_outbrain' stream for marketer {current_marketer_id}.")
        
    return state


def sync_link_performance(state, access_token, account_id, campaign_id,
                          link_id, config):
    return sync_performance(
        state,
        access_token,
        account_id,
        'link_performance_outbrain',
        link_id,
        {'promotedLinkId': link_id},
        {'campaignId': campaign_id,
         'linkId': link_id,
         'account_id': account_id},
        config)

def sync_performance(state, access_token, account_id, table_name, state_sub_id,
                     extra_params, extra_persist_fields, config):
    """
    This function is heavily parameterized as it is used to sync performance
    both based on campaign ID alone, and by campaign ID and link ID.

    - `state`: state map
    - `access_token`: access token for Outbrain Amplify API
    - `account_id`: Outbrain marketer ID
    - `table_name`: the table name to use. At present, one of
                    `campaign_performance` or `link_performance`.
    - `state_sub_id`: the id to use within the state map to identify this
                      sub-object.
    - `extra_params`: extra params sent to the Outbrain API
    - `extra_persist_fields`: extra fields pushed into the destination data.
    - `config`: configuration dictionary containing start_date and end_date
    """
    from_date = datetime.datetime.strptime(
        state.get(table_name, {})
             .get(state_sub_id, DEFAULT_START_DATE),
        '%Y-%m-%d').date() - datetime.timedelta(days=0)

    # Use end_date from config if provided, otherwise use today's date
    to_date = datetime.datetime.strptime(config.get('end_date'), '%Y-%m-%d').date() \
              if config.get('end_date') \
              else datetime.date.today()

    interval_in_days = 100

    date_ranges = get_date_ranges(from_date, to_date, interval_in_days)

    last_request_start = None

    for date_range in date_ranges:
        logger.info(
            'Pulling {} for {} from {} to {}'
            .format(table_name,
                    extra_persist_fields,
                    date_range.get('from_date'),
                    date_range.get('to_date')))

        params = {
            'from': date_range.get('from_date'),
            'to': date_range.get('to_date'),
            'limit': 200
        }
        params.update(extra_params)

        last_request_start = time.time()
        response = request(
            '{}/reports/marketers/{}/campaigns'.format(BASE_URL, account_id),
            access_token,
            params)
        last_request_end = time.time()

        logger.info('Done in {} sec'.format(time.time() - last_request_start))

        performance = [
            parse_performance(result, extra_persist_fields)
            for result in response.json().get('results')]

        singer.write_records(table_name, performance)

        if performance:
            last_record = performance[-1]
            new_from_date = last_record.get('fromDate')

            state[table_name][state_sub_id] = new_from_date
            singer.write_state(state)

        if last_request_start is not None and \
           (time.time() - last_request_end) < 30:
            to_sleep = 30 - (time.time() - last_request_end)
            logger.info(
                'Limiting to 2 requests per minute. Sleeping {} sec '
                'before making the next reporting request.'
                .format(to_sleep))
            time.sleep(to_sleep)

def parse_campaign(campaign, account_id):
    """Parse campaign data while preserving Decimal types."""
    logger.info("Raw campaign data - checking:")
    logger.info(json.dumps(campaign, default=str, indent=2))
    
    # Handle budget separately but preserve Decimal types
    budget = None
    if campaign.get('budget'):
        try:
            budget_raw = campaign['budget']
            budget = {
                'id': str(budget_raw.get('id', '')),
                'name': str(budget_raw.get('name', '')),
                'amount': budget_raw.get('amount'),
                'currency': str(budget_raw.get('currency', '')),
                'creationTime': parse_datetime(budget_raw.get('creationTime')),
                'lastModified': parse_datetime(budget_raw.get('lastModified')),
                'type': str(budget_raw.get('type', ''))
            }
            logger.info("Budget after processing:")
            logger.info(json.dumps(budget, default=str, indent=2))
        except Exception as e:
            logger.error(f"Error processing budget: {e}")
            logger.error(f"Raw budget data: {json.dumps(campaign['budget'], default=str)}")
    # Build result with explicit type casting
    result = {
        'id': str(campaign.get('id', '')),  # Ensure string type
        'name': str(campaign.get('name', '')),
        'campaignOnAir': bool(campaign.get('liveStatus', {}).get('campaignOnAir', False)),
        'onAirReason': str(campaign.get('liveStatus', {}).get('onAirReason', '')),
        'enabled': bool(campaign.get('enabled', False)),
        'budget': budget,
        'cpc': campaign.get('cpc'),
        'currency': str(campaign.get('currency', 'USD')),  # Add missing fields, starting from 'currency'
        'status': str(campaign.get('status', '')),
        'startDate': parse_datetime(campaign.get('startDate')) or None,
        'endDate': parse_datetime(campaign.get('endDate')) or None,
        'account_id': str(account_id)
    }

    logger.info("Processed campaign data - checking:")
    logger.info(json.dumps(result, default=str, indent=2))

    return result

def sync_campaigns(state, access_token, account_id, config):
    logger.info('Syncing campaigns.')

    account_name = ""
    try:
        marketers_response = request(
            '{}/marketers'.format(BASE_URL),
            access_token)
        
        marketers = marketers_response.json().get('marketers', [])
        for marketer in marketers:
            if marketer.get('id') == account_id:
                account_name = marketer.get('name', '')
                logger.info(f"Found account name: {account_name} for account ID: {account_id}")
                break
    except Exception as e:
        logger.error(f"Error fetching account name: {e}")

    start = time.time()
    response = request(
        '{}/marketers/{}/campaigns'.format(BASE_URL, account_id),
        access_token, {})

    raw_campaigns = response.json().get('campaigns', [])
    logger.info(f"Got {len(raw_campaigns)} campaigns from API")
    
    campaigns = []
    for campaign in raw_campaigns:
        try:
            # Pass account_id to parse_campaign
            parsed = parse_campaign(campaign, account_id)
            campaigns.append(parsed)
            
            # Sync performance data for each campaign
            campaign_id = campaign.get('id')
            if campaign_id:
                logger.info(f"Syncing performance data for campaign {campaign_id}")
                sync_campaign_performance(state, access_token, account_id, campaign_id, account_name, config)

                
        except Exception as e:
            logger.error(f"Error parsing campaign: {e}")
            logger.error(f"Problematic campaign data: {json.dumps(campaign, default=str)}")

    logger.info("About to write records")
    singer.write_records('campaigns_outbrain', campaigns)


def parse_link(link, account_id):
    """Parse a promoted link record with decimal handling."""
    # Convert all Decimals first
    link = decimal_to_float(link)
    
    # Handle datetime fields
    link['creationTime'] = parse_datetime(link.get('creationTime'))
    link['lastModified'] = parse_datetime(link.get('lastModified'))
    
    # Ensure all string fields are properly encoded
    return {
        'id': str(link.get('id')),
        'name': str(link.get('name', '')),
        'url': str(link.get('url', '')),
        'creationTime': link.get('creationTime'),
        'lastModified': link.get('lastModified'),
        'isActive': bool(link.get('isActive', False)),
        'cpc': float(link.get('cpc', 0.0)),
        'campaignId': str(link.get('campaignId', '')),
        'content': decimal_to_float(link.get('content', {})),
        'approvalStatus': str(link.get('approvalStatus', '')),
        'thumbnailUrl': str(link.get('thumbnailUrl', '')),
        'isMarkedForRemoval': bool(link.get('isMarkedForRemoval', False)),
        'account_id': str(account_id)
    }

def sync_links(state, access_token, account_id, campaign_id, config):
    processed_count = 0
    total_count = -1
    fully_synced_count = 0
    limit = 100

    while processed_count != total_count:
        logger.info(
            'Syncing {} links for campaign {} starting from offset {}'
            .format(limit,
                    campaign_id,
                    processed_count))

        start = time.time()
        response = request(
            '{}/campaigns/{}/promotedLinks'.format(BASE_URL, campaign_id),
            access_token, {
                'limit': 100,
                'offset': processed_count
            })

        # Pass account_id to parse_link
        links = [parse_link(link, account_id) for link
                 in response.json().get('promotedLinks', [])]

        singer.write_records('links_outbrain', links)

        total_count = response.json().get('totalCount')
        processed_count = processed_count + len(links)

        for link in links:
            logger.info(
                'Syncing link performance for link {} of {}.'.format(
                    fully_synced_count,
                    total_count))

            sync_link_performance(state, access_token, account_id, campaign_id,
                                  link.get('id'), config)

            fully_synced_count = fully_synced_count + 1

        logger.info('Done in {} sec, processed {} of {} links.'
                    .format(time.time() - start,
                            processed_count,
                            total_count))

    logger.info('Done syncing links for campaign {}.'.format(campaign_id))

def validate_config(config):
    required_keys = ['username', 'password', 'start_date'] 
    missing_keys = []
    null_keys = []
    has_errors = False

    for required_key in required_keys:
        if required_key not in config:
            missing_keys.append(required_key)
        elif config.get(required_key) is None:
            null_keys.append(required_key)

    if missing_keys:
        logger.fatal("Config is missing keys: {}".format(", ".join(missing_keys)))
        has_errors = True

    if null_keys:
        logger.fatal("Config has null keys: {}".format(", ".join(null_keys)))
        has_errors = True

    # Validate end_date format if provided
    if config.get('end_date'):
        try:
            datetime.datetime.strptime(config['end_date'], '%Y-%m-%d')
        except ValueError:
            logger.fatal("end_date must be in YYYY-MM-DD format")
            has_errors = True
    
    # Validate start_date format
    try:
        datetime.datetime.strptime(config['start_date'], '%Y-%m-%d')
    except ValueError:
        logger.fatal("start_date must be in YYYY-MM-DD format")
        has_errors = True

    if has_errors:
        raise RuntimeError("Config validation failed.")


def do_sync(args):
    global DEFAULT_START_DATE
    logger.info("Starting Outbrain tap sync")

    with open(args.config) as config_file:
        config = json.load(config_file)
    validate_config(config)

    state = DEFAULT_STATE
    if args.state:
        logger.info(f"Loading state from {args.state}")
        with open(args.state) as state_file:
            state = json.load(state_file)
    else:
        logger.info("No state file provided, using default state.")

    username = config['username']
    password = config['password']
    
    DEFAULT_START_DATE = config['start_date'][:10] 

    access_token = config.get('access_token')
    if not access_token:
        logger.info("Access token not found in config, generating new one.")
        access_token = generate_token(username, password)

    if not access_token:
        logger.fatal("Failed to obtain access token. Exiting.")
        raise RuntimeError("Access token generation failed.")
    logger.info("Successfully obtained access token.")

    # Schema
    logger.info("Writing schema for 'campaign_performance_outbrain'")
    singer.write_schema('campaign_performance_outbrain',
                        schemas.campaign_performance,
                        key_properties=[]) 

    # # singer.write_schema('campaigns_outbrain', schemas.campaign, key_properties=["id"])

    # Fetch all marketers associated with the token
    all_marketers = get_all_marketers(access_token)

    if not all_marketers:
        logger.warning("No marketers found or accessible with the provided token. No campaign performance data will be synced.")
    else:
        logger.info(f"Found {len(all_marketers)} marketers. Syncing campaign performance for each.")
        for marketer_info in all_marketers:
            current_marketer_id = marketer_info['id']
            current_marketer_name = marketer_info['name']
            
            # Call the sync function for campaign performance for the current marketer
            logger.info(f"Processing marketer: {current_marketer_name} (ID: {current_marketer_id})")
            sync_campaign_performance(state, access_token, current_marketer_id, current_marketer_name, config)

    # # sync_campaigns(state, access_token, config['account_id'], config) # Original call might have used account_id from config

    logger.info("Outbrain tap sync completed.")
    # singer.write_state(state) # Uncomment if state is modified and needs to be persisted.

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config', help='Config file', required=True)
    parser.add_argument(
        '-s', '--state', help='State file')

    args = parser.parse_args()

    try:
        do_sync(args)
    except RuntimeError:
        logger.fatal("Run failed.")
        exit(1)

if __name__ == '__main__':
    main()
