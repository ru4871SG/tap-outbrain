link = {
    'type': 'object',
    'properties': {
        'id': {
            'type': 'string',
            'maxLength': 191,
            'description': ('ID of this PromotedLink, i.e. '
                            '"00f4b02153ee75f3c9dc4fc128ab041962"')
        },
        'account_id': {
            'type': 'string',
            'description': 'Outbrain Account ID'
        },
        'campaignId': {
            'type': 'string',
            'description': ('The ID of the campaign to which the '
                            'PromotedLink belongs, i.e. '
                            '"00f4b02153ee75f3c9dc4fc128ab041963"')
        },
        'text': {
            'type': 'string',
            'description': ('The text of the PromotedLink, i.e. "Google to '
                            'take over huge NASA hangar, give execs\' private '
                            'planes a home"'),
        },
        'lastModified': {
            'type': 'string',
            'format': 'date-time',
            'description': ('The time when the PromotedLink was last '
                            'modified, i.e. "2013-03-16T10:32:31Z"')
        },
        'creationTime': {
            'type': 'string',
            'format': 'date-time',
            'description': ('The time when the PromotedLink was created, '
                            'i.e. "2013-01-14T07:19:16Z"')
        },
        'url': {
            'type': 'string',
            'description': ('The URL visitors will be sent to upon clicking '
                            'the PromotedLink, i.e. "http://www.engadget.com'
                            '/2014/02/11/nasa-google-hangar-one/"')
        },
        'siteName': {
            'type': 'string',
            'description': ('The name of the publisher the PromotedLink '
                            'URL points to, i.e. "cnn.com"')
        },
        'sectionName': {
            'type': 'string',
            'description': ('The section name of the site the PromotedLink '
                            'URL points to, i.e. "Sports"')
        },
        'status': {
            'type': 'string',
            'description': ('The review status of the PromotedLink, '
                            'i.e. "PENDING"')
        },
        'cachedImageUrl': {
            'type': 'string',
            'description': ('The URL of the PromotedLink\'s image, cached '
                            'on Outbrain\'s servers, i.e. "http://images'
                            '.outbrain.com/imageserver/v2/s/gtE/n/plcyz/abc'
                            '/iGYzT/plcyz-f8A-158x110.jpg"')
        },
        'enabled': {
            'type': 'boolean',
            'description': ('Designates whether this PromotedLink will be '
                            'served.')
        },
        'archived': {
            'type': 'boolean',
            'description': ('Designates whether this PromotedLink is '
                            'archived.')
        },
        'documentLanguage': {
            'type': 'string',
            'description': ('The 2-letter code for the language of this '
                            'PromotedLink (via the PromotedLinks URL), '
                            'i.e. "EN"')
        },
        'cpc': {
            'type': 'number',
            'description': ('Cost per click, i.e. 0.58')
        }
    }
}


campaign = {
    'type': 'object',
    'properties': {
        'id': {
            'type': 'string',
            'description': 'Campaign ID'
        },
        'name': {
            'type': 'string',
            'description': 'Campaign name'
        },
        'account_id': {
            'type': 'string',
            'description': 'Outbrain Account ID'
        },
        'account_name': {
            'type': 'string',
            'description': 'Outbrain Account Name'
        },
        'campaignOnAir': {
            'type': 'boolean',
            'description': ('Is the campaign on air, same as campaignOnAir '
                            'in Live Status')
        },
        'onAirReason': {
	    'type': 'string',
            'description': ('The reason for the campaign on air status, same '
                            'as onAirReason in Live Status')
        },
        'enabled': {
            'type': 'boolean',
            'description': 'Is the campaign enabled'
        },
        'budget': {
            'type': 'object',
            'description': ('Partial Budget entity of a campaign. For full '
                            'details use Budget'),
            'properties': {
                'id': {
	            'type': 'string',
                    'description': ('The id of this Budget, i.e. '
                                    '"00f4b02153ee75f3c9dc4fc128ab041962"')
                },
                'name': {
                    'type': 'string',
                    'description': ('The name of this Budget, i.e. '
                                    '"First quarter budget"'),
                },
                'shared': {
                    'type': 'boolean',
                    'description': ('Whether the Budget is shared between '
                                    'Campaigns, provided for convenience '
                                    'based on the number of Campaigns '
                                    'associated to this Budget, i.e. true')
                },
                'amount': {
                    'type': 'number',
                    'description': ('The monetary amount of this Budget, '
                                    'i.e. 2000.00')
                },
                'currency': {
                    'type': 'string',
                    'description': ('The currency denomination applied to the '
                                    'budget amount, i.e. "USD"')
                },
                'amountRemaining': {
                    'type': 'number',
                    'description': ('The unspent monetary amount remaining on '
                                    'this Budget, i.e. 150.00')
                },
                'amountSpent': {
                    'type': 'number',
                    'description': ('The spent monetary amount of this '
                                    'Budget, i.e. 1850.00')
                },
                'creationTime': {
                    'type': 'string',
                    'format': 'date-time',
                    'description': ('The time when this Budget was created, '
                                    'i.e. "2013-01-14 07:19:16"')
                },
                'lastModified': {
                    'type': 'string',
                    'format': 'date-time',
                    'description': ('The last modification date of this '
                                    'Budget, i.e. "2014-01-15 12:24:01"')
                },
                'startDate': {
                    'type': ['string', 'null'],  # Allow both string and null
                    'format': 'date',
                    'description': ('The date this Budget is scheduled to '
                                    'begin spending, i.e. "2014-01-15"')
                },
                'endDate': {
                    'type': ['string', 'null'],  # Allow both string and null
                    'format': 'date',
                    'description': ('The date this Budget is scheduled to '
                                    'stop spending. If runForever is true '
                                    'this will not be used. i.e. "2014-01-17"')
                },
                'runForever': {
                    'type': 'boolean',
                    'description': ('Designates whether the budged has an end '
                                    'date In case of true, "endDate" '
                                    'attribute will not be part of the '
                                    'Budgets attributes. i.e. true')
                },
                'type': {
                    'type': 'string',
                    'description': ('Controls on which period the Budget '
                                    'refreshes, i.e. "MONTHLY"')
                },
                'pacing': {
                    'type': 'string',
                    'description': ('Controls how fast the Budget will be '
                                    'spent, i.e. "AUTOMATIC"')
                },
                'dailyTarget': {
                    'type': 'number',
                    'description': ('The maximum amount of spend that is '
                                    'allowed per day. Relevant for '
                                    'DAILY_TARGET pacing. i.e. 100.00')
                },
                'maximumAmount': {
                    'type': 'number',
                    'description': ('The maximum amount allowed if defined, '
                                    'i.e. 100.00')
                }
            }
        },
        'cpc': {
            'type': 'number',
            'description': ('Cost per monetized user action (for example '
                            'cost per click). See Currencies for valid '
                            'cost values')
        },
        'currency': {
            'type': 'string',
            'description': 'The currency used for the campaign, default is USD'
        },
        'status': {
            'type': 'string',
            'description': 'The current status of the campaign'
        },
        'startDate': {
            'type': ['string', 'null'],  # Allow both string and null
            'format': 'date',
            'description': 'The date when the campaign starts'
        },
        'endDate': {
            'type': ['string', 'null'],  # Allow both string and null
            'format': 'date',
            'description': 'The date when the campaign ends'
        }
    }
}


campaign_performance = {
    'type': 'object',
    'properties': {
        'marketer_id': {'type': ['null', 'string'], 'description': 'Marketer ID from Outbrain'},
        'marketer_name': {'type': ['null', 'string'], 'description': 'Marketer Name from Outbrain'},
        'campaign_id': {'type': ['null', 'string'], 'description': 'Campaign ID'},
        'campaign_name': {'type': ['null', 'string'], 'description': 'Campaign Name'},
        'impressions': {'type': ['null', 'integer'], 'description': 'Total impressions'},
        'clicks': {'type': ['null', 'integer'], 'description': 'Total clicks'},
        'totalConversions': {'type': ['null', 'integer'], 'description': 'Total conversions (including view-through)'},
        'conversions': {'type': ['null', 'integer'], 'description': 'Click-through conversions'},
        'viewConversions': {'type': ['null', 'integer'], 'description': 'View-through conversions'},
        'spend': {'type': ['null', 'number'], 'description': 'Total spend'},
        'ecpc': {'type': ['null', 'number'], 'description': 'Effective Cost Per Click'},
        'ctr': {'type': ['null', 'number'], 'description': 'Click-Through Rate'},
        'dstFeeCost': {'type': ['null', 'number'], 'description': 'DST Fee Cost'},
        'conversionRate': {'type': ['null', 'number'], 'description': 'Conversion Rate (click-through)'},
        'viewConversionRate': {'type': ['null', 'number'], 'description': 'View-through Conversion Rate'},
        'cpa': {'type': ['null', 'number'], 'description': 'Cost Per Acquisition (click-through)'},
        'totalCpa': {'type': ['null', 'number'], 'description': 'Total Cost Per Acquisition'},
        'totalSumValue': {'type': ['null', 'number'], 'description': 'Total sum of conversion values'},
        'sumValue': {'type': ['null', 'number'], 'description': 'Sum of click-through conversion values'},
        'viewSumValue': {'type': ['null', 'number'], 'description': 'Sum of view-through conversion values'},
        'totalAverageValue': {'type': ['null', 'number'], 'description': 'Total average conversion value'},
        'averageValue': {'type': ['null', 'number'], 'description': 'Average click-through conversion value'},
        'viewAverageValue': {'type': ['null', 'number'], 'description': 'Average view-through conversion value'},
        'totalRoas': {'type': ['null', 'number'], 'description': 'Total Return On Ad Spend'},
        'roas': {'type': ['null', 'number'], 'description': 'Return On Ad Spend (click-through)'},
        'optimizedConversions': {'type': ['null', 'integer'], 'description': 'Optimized conversions count'},
        'fetched_from': {'type': ['null', 'string'], 'format': 'date', 'description': 'Start date of the report period'},
        'fetched_to': {'type': ['null', 'string'], 'format': 'date', 'description': 'End date of the report period'}
    }
}

link_performance = {
    'type': 'object',
    'properties': {
        'campaignId': {
            'type': 'string',
            'description': ('The campaign ID for this record.')
        },
        'linkId': {
            'type': 'string',
            'description': ('The link ID for this record.')
        },
        'account_id': {
            'type': 'string',
            'description': 'Outbrain Account ID'
        },
        'fromDate': {
            'type': 'string',
            'format': 'date',
            'description': 'The start date for this record.'
        },
        'impressions': {
            'type': 'number',
            'description': ('Total number of PromotedLinks impressions across '
                            'the entire query range.'),
        },
        'clicks': {
            'type': 'number',
            'description': ('Total PromotedLinks clicks across the entire '
                            'query range.'),
        },
        'ctr': {
            'type': 'number',
            'description': ('The average CTR (Click Through Rate) percentage '
                            'across the entire query range (clicks / '
                            'impressions)/100.'),
        },
        'spend': {
            'type': 'number',
            'description': ('The total amount of money spent across the '
                            'entire query range.'),
        },
        'ecpc': {
            'type': 'number',
            'description': ('The effective (calculated) average CPC (Cost Per '
                            'Click) across the entire query range. '
                            'Calculated as: (spend / clicks)'),
        },
        'conversions': {
            'type': 'number',
            'description': ('The total number of conversions calculated '
                            'across the entire query range.')
        },
        'conversionRate': {
            'type': 'number',
            'description': ('The average rate of conversions per click '
                            'percentage across the entire query range. '
                            'Calculated as: (conversions / clicks)/100')
        },
        'cpa': {
            'type': 'number',
            'description': ('The average CPA (Cost Per Acquisition) '
                            'calculated across the entire query range. '
                            'Calculated as: (spend / conversions)')
        }
    }
}
