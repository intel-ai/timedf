import re
import os
import sys
import errno
import shutil

HEADERS = [
    "loan_id,orig_channel,seller_name,orig_interest_rate,orig_upb,orig_loan_term,orig_date,first_pay_date,orig_ltv,orig_cltv,num_borrowers,dti,borrower_credit_score,first_home_buyer,loan_purpose,property_type,num_units,occupancy_status,property_state,zip,mortgage_insurance_percent,product_type,coborrow_credit_score,mortgage_insurance_type,relocation_mortgage_indicator,year_quarter_ignore",
    "loan_id,monthly_reporting_period,servicer,interest_rate,current_actual_upb,loan_age,remaining_months_to_legal_maturity,adj_remaining_months_to_maturity,maturity_date,msa,current_loan_delinquency_status,mod_flag,zero_balance_code,zero_balance_effective_date,last_paid_installment_date,foreclosed_after,disposition_date,foreclosure_costs,prop_preservation_and_repair_costs,asset_recovery_costs,misc_holding_expenses,holding_taxes,net_sale_proceeds,credit_enhancement_proceeds,repurchase_make_whole_proceeds,other_foreclosure_proceeds,non_interest_bearing_upb,principal_forgiveness_upb,repurchase_make_whole_proceeds_flag,foreclosure_principal_write_off_amount,servicing_activity_indicator",
]


def main():
    headers = {h.count(","): h for h in HEADERS}
    try:
        source, dest = [os.path.abspath(x) for x in sys.argv[1:]]
    except TypeError:
        raise TypeError("Usage: %s source-path dest-path" % sys.argv[0])

    for root, _, files in os.walk(source):
        if not files:
            continue
        targetDir = os.path.join(dest, os.path.relpath(root, source))
        try:
            os.makedirs(targetDir)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise
        for fn in files:
            srcFile = os.path.join(root, fn)
            destFile = os.path.join(targetDir, fn)
            if not re.match(r".*\d+Q\d\..*", fn):
                print("Copying %s as is..." % fn)
                shutil.copy(srcFile, destFile)
                continue

            print("Processing %s...\n\tReading" % srcFile)
            with open(srcFile) as inp:
                data = inp.read()
            print("\tReplacing delimiters")
            data = data.replace(",", "_").replace("|", ",")
            line = data[: data.find("\n")].strip()
            try:
                header = headers[line.count(",")]
            except KeyError:
                print("Warning: cannot determine headers for %s" % srcFile)
                header = None
            print("\tTransforming dates")
            data = re.sub(r"(?<=,)(\d+)/(\d+)/(\d+)(?=,)", r"\3-\1-\2", data)
            data = re.sub(r"(?<=,)(\d+)/(\d+)(?=,)", r"\2-\1-01", data)
            print("\tWriting output to %s" % destFile)
            with open(destFile, "w") as out:
                if header:
                    out.write(header + "\n")
                out.write(data)


if __name__ == "__main__":
    main()
