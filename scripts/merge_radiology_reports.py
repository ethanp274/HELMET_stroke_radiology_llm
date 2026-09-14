
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime


dt_parser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") 

def read_csv_file(filename):
        path = os.path.join("data/raw", filename)
        return pd.read_csv(path, parse_dates=True, date_format=dt_parser)

def edit_mgh():
        mgh_ltv = read_csv_file("LTV_2.0.csv")
        mgh_text = read_csv_file("MGH_rad_reports_text.csv")
        mgh_text.loc[:,'ptid'] = mgh_text.loc[:,'ptid'].ffill()
        mgh_text = mgh_text[mgh_text['rad_report_text'].notna()]

        mgh_text.rename(columns = {'reportdt':'dt', 'rad_report_text':'rad_text'}, inplace = True)
        mgh_text.drop(['hospital', 'reportseq'], inplace = True, axis = 1)

        mgh_ltv['dt_match'] = pd.to_datetime(mgh_ltv['dt'].copy(), utc = True)
        mgh_text['dt_match'] = pd.to_datetime(mgh_text['dt'].copy(), utc = True)

        mgh_ltv.set_index(['ptid', 'dt_match'], inplace = True)
        mgh_text.set_index(['ptid', 'dt_match'], inplace = True)

        mgh_ltv_text = pd.merge(left = mgh_ltv, right = mgh_text, how = 'left', on = ['ptid', 'dt_match'])
        mgh_ltv_text.drop(['dt_y', 'dt_match'], inplace = True)
        mgh_ltv_text.rename(columns = {'dt_x':'dt'}, inplace = True)

        mgh_ltv_text.reset_index(drop = False, inplace = True)

        mgh_ltv_text.to_json("data/processed/mgh_ltv_2.0_text.json")

def edit_bmc():
        bmc_ltv = read_csv_file('LTV_BMC_2.csv')
        bmc_text = read_csv_file('BMC_rad_reports_text.csv')
        print(bmc_text.columns)

        bmc_text.drop(
                ['mrn', 'redcap_repeat_instrument', 'redcap_repeat_instance', 'lsw', 'presdt', 'masterid', 'cdwid', 'mrn_repeat', 'size_mls', 'size_max_pgs', 'radreport_textdatetime', 'radreport_24', 'ndceltme_radrep', 'ndceltme_raddt', 'comments'],
                inplace = True,
                axis = 1
        )
        bmc_text.rename(columns = {'id': 'ptid', 'imagedt':'dt', 'rc_mls': 'size_mls', 'rc_pgs':'size_pgs', 'rad_rep':'rad_text'}, inplace = True)
        bmc_text = bmc_text[bmc_text['rad_text'].notna()]
        bmc_text_only = bmc_text.drop(['size_mls', 'size_pgs'], axis = 1, inplace = False).copy()
        
        bmc_ltv['dt_match'] = pd.to_datetime(bmc_ltv['dt'].copy())
        bmc_ltv.rename(columns = {'redcap_id':'ptid'}, inplace = True)
        bmc_text_only['dt_match'] = pd.to_datetime(bmc_text_only['dt'].copy())

        bmc_ltv.set_index(['ptid', 'dt_match'], inplace = True)
        bmc_text_only.set_index(['ptid', 'dt_match'], inplace = True)

        bmc_ltv_text = pd.merge(left = bmc_ltv, right = bmc_text_only, how = 'left', on = ['ptid', 'dt_match'])
        bmc_ltv_text.drop(['dt_y', 'dt_match'], inplace = True)
        bmc_ltv_text.rename(columns = {'dt_x':'dt'}, inplace = True)

        bmc_ltv_text.reset_index(drop = False, inplace = True)

        bmc_ltv_text.to_json("data/processed/bmc_ltv_2.0_text.json")


if __name__ == "__main__":
        
        edit_bmc()
