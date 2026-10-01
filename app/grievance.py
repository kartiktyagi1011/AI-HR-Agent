import re
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class GrievanceModel:

    def __init__(self, csv_path="data/grievance_dataset.csv"):

        self.df = pd.read_csv(csv_path)
        self.df = self.df.fillna("")

        # ---------------------------------------------------------
        # TRAIN ML MODELS
        # ---------------------------------------------------------

        self.category_model = self._train("Category")
        self.severity_model = self._train("Severity")
        self.department_model = self._train("Department")

    # =========================================================
    # TRAIN ML MODEL
    # =========================================================

    def _train(self, target):

        model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True
                )
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            )
        ])

        model.fit(
            self.df["Incident_Summary"].astype(str),
            self.df[target].astype(str)
        )

        return model

    # =========================================================
    # CATEGORY CLASSIFICATION
    # =========================================================

    def classify_category(self, text):

        text = text.lower().strip()

        # ---------------------------------------------------------
        # PAYROLL / COMPENSATION
        # ---------------------------------------------------------

        payroll_words = [
            "salary",
            "payroll",
            "wage",
            "payment",
            "paycheck",
            "credited",
            "credit salary",
            "salary credited",
            "salary not received",
            "salary not credited",
            "salary not paid",
            "salary has not been credited",
            "salary has not been paid",
            "bonus",
            "commission",
            "deduction",
            "payslip",
            "pay slip",
            "overtime payment",
            "overtime pay",
            "compensation",
            "benefit payment"
        ]

        if any(word in text for word in payroll_words):
            return "Compensation & Benefits Discrepancy"

        # ---------------------------------------------------------
        # LEAVE / TIME OFF
        # ---------------------------------------------------------

        leave_words = [
            "leave",
            "vacation",
            "pto",
            "paid time off",
            "sick leave",
            "casual leave",
            "holiday",
            "maternity",
            "paternity",
            "parental leave",
            "leave balance",
            "annual leave",
            "time off",
            "leave request",
            "leave approval",
            "leave denied",
            "leave rejected"
        ]

        if any(word in text for word in leave_words):
            return "Work-Life Balance & Leave Rights"

        # ---------------------------------------------------------
        # HARASSMENT / DISCRIMINATION
        # ---------------------------------------------------------

        harassment_words = [
            "harassment",
            "harassed",
            "discrimination",
            "discriminated",
            "sexual harassment",
            "sexual misconduct",
            "bullying",
            "inappropriate comments",
            "inappropriate comment",
            "inappropriate behavior",
            "inappropriate behaviour",
            "inappropriate remarks",
            "offensive comments",
            "offensive comment",
            "offensive remarks",
            "offensive behavior",
            "offensive behaviour",
            "verbal abuse"
        ]

        if any(word in text for word in harassment_words):
            return "Harassment & Discrimination"

        # ---------------------------------------------------------
        # RETALIATION
        # ---------------------------------------------------------

        retaliation_words = [
             "retaliation",
             "retaliated",
             "retaliate",
             "retaliatory",
             "revenge",
             "reprisal",

             "punished for reporting",
            "punished after complaint",
             "punished for complaining",
            "punished after reporting",

             "threatened after reporting",
             "threatened because i reported",
             "threatened to punish me after i reported",
             "threatened to punish me for reporting",
             "threatened to punish after i reported",
             "threatened because of my complaint",
             "threatened after my complaint",

            "negative action after complaint",
            "negative action after reporting"
        ]

        if any(word in text for word in retaliation_words):
           return "Retaliation Concerns"

        # ---------------------------------------------------------
        # HOSTILE WORK ENVIRONMENT
        # ---------------------------------------------------------

        hostile_words = [
            "hostile",
            "hostile workplace",
            "hostile work environment",
            "toxic workplace",
            "toxic work environment",
            "toxic environment",
            "intimidation",
            "threatening workplace",
            "fearful workplace",
            "unsafe work environment"
        ]

        if any(word in text for word in hostile_words):
            return "Hostile Work Environment"

        # ---------------------------------------------------------
        # WORKPLACE SAFETY
        # ---------------------------------------------------------

        safety_words = [
            "unsafe",
            "safety",
            "accident",
            "injury",
            "fire hazard",
            "workplace hazard",
            "hazard",
            "ergonomic",
            "broken chair",
            "unsafe equipment",
            "unsafe workstation",
            "safety equipment",
            "workplace injury"
        ]

        if any(word in text for word in safety_words):
            return "Workplace Safety & Ergonomics"

        # ---------------------------------------------------------
        # REMOTE WORK / EQUIPMENT
        # ---------------------------------------------------------

        remote_words = [
            "work from home",
            "wfh",
            "remote work",
            "remote working",
            "home working",
            "remote",
            "laptop for remote work",
            "remote equipment",
            "company laptop",
            "work laptop",
            "laptop not working",
            "computer not working",
            "monitor not working",
            "keyboard not working",
            "mouse not working",
            "vpn",
            "remote access"
        ]

        if any(word in text for word in remote_words):
            return "Remote Work & Equipment Policy"

        # ---------------------------------------------------------
        # POLICY NON-COMPLIANCE
        # ---------------------------------------------------------

        policy_words = [
            "policy violation",
            "policy breach",
            "policy violated",
            "policy violation",
            "non-compliance",
            "noncompliance",
            "not following policy",
            "violated the policy",
            "breached the policy",
            "policy was ignored"
        ]

        if any(word in text for word in policy_words):
            return "Policy Non-Compliance"

        # ---------------------------------------------------------
        # MANAGERIAL MISCONDUCT
        # ---------------------------------------------------------

        manager_words = [
            "manager misconduct",
            "abuse of authority",
            "abuse of power",
            "manager abused",
            "supervisor abused",
            "manager treated me unfairly",
            "manager is unfair",
            "manager misconduct",
            "supervisor misconduct",
            "manager ignored my complaint",
            "manager refused",
            "manager denied"
        ]

        if any(word in text for word in manager_words):
            return "Managerial Misconduct"

        # ---------------------------------------------------------
        # MANAGER-SPECIFIC CASES
        # ---------------------------------------------------------

        # Only classify as managerial misconduct when the text
        # clearly indicates a management-related issue.
        manager_context = [
            "my manager",
            "my supervisor",
            "my boss",
            "manager's behavior",
            "manager's behaviour",
            "supervisor's behavior",
            "supervisor's behaviour"
        ]

        misconduct_context = [
            "unfair",
            "abuse",
            "ignored",
            "refused",
            "denied",
            "misconduct",
            "favoritism",
            "favouritism",
            "inappropriate"
        ]

        if (
            any(word in text for word in manager_context)
            and any(word in text for word in misconduct_context)
        ):
            return "Managerial Misconduct"

        # ---------------------------------------------------------
        # ML FALLBACK
        # ---------------------------------------------------------

        return self.category_model.predict([text])[0]

    # =========================================================
    # SEVERITY CLASSIFICATION
    # =========================================================

    def classify_severity(self, text, category):

        text = text.lower().strip()

        # ---------------------------------------------------------
        # CRITICAL
        # ---------------------------------------------------------

        critical_words = [
            "immediate danger",
            "life threatening",
            "life-threatening",
            "serious injury",
            "severe injury",
            "physical violence",
            "violent attack",
            "death threat",
            "sexual assault",
            "assault",
            "fire emergency",
            "emergency",
            "danger to life",
            "risk to life",
            "imminent danger"
        ]

        if any(word in text for word in critical_words):
            return "Critical"

        # ---------------------------------------------------------
        # HIGH
        # ---------------------------------------------------------

        high_words = [
            "harassment",
            "harassed",
            "discrimination",
            "discriminated",
            "sexual harassment",
            "bullying",
            "retaliation",
            "retaliated",
            "retaliatory",

            "salary not paid",
            "salary not credited",
            "salary has not been credited",
            "salary has not been paid",
            "salary has not",
            "not been credited",
            "not been paid",

            "major financial loss",
            "major financial",
            "fraud",

            "repeated harassment",
            "repeatedly harassed",
            "repeated bullying",
            "repeated discrimination",

            "threatened",
            "intimidation",
            "hostile work environment",
            "hostile workplace"
        ]

        if any(word in text for word in high_words):
            return "High"

        # ---------------------------------------------------------
        # CATEGORY-BASED SEVERITY
        # ---------------------------------------------------------

        if category == "Harassment & Discrimination":

            # Harassment/discrimination cases are treated as
            # high priority unless a critical condition was
            # already detected above.
            return "High"

        if category == "Retaliation Concerns":
            return "High"

        # ---------------------------------------------------------
        # MEDIUM SIGNALS
        # ---------------------------------------------------------

        medium_words = [
            "repeatedly",
            "repeated",
            "ongoing",
            "multiple times",
            "significant",
            "serious concern",
            "financial loss",
            "denied",
            "refused",
            "broken equipment",
            "cannot work",
            "unable to work",
            "work is affected"
        ]

        if any(word in text for word in medium_words):
            return "Medium"

        # ---------------------------------------------------------
        # ML FALLBACK
        # ---------------------------------------------------------

        return self.severity_model.predict([text])[0]

    # =========================================================
    # DEPARTMENT / ROUTING
    # =========================================================

    def classify_department(self, text, category):

        text = text.lower().strip()

        # ---------------------------------------------------------
        # COMPENSATION
        # ---------------------------------------------------------

        if category == "Compensation & Benefits Discrepancy":
            return "Finance / HR"

        # ---------------------------------------------------------
        # LEAVE
        # ---------------------------------------------------------

        if category == "Work-Life Balance & Leave Rights":
            return "Human Resources"

        # ---------------------------------------------------------
        # HARASSMENT
        # ---------------------------------------------------------

        if category == "Harassment & Discrimination":
            return "Human Resources"

        # ---------------------------------------------------------
        # RETALIATION
        # ---------------------------------------------------------

        if category == "Retaliation Concerns":
            return "HR / Compliance"

        # ---------------------------------------------------------
        # REMOTE WORK / EQUIPMENT
        # ---------------------------------------------------------

        if category == "Remote Work & Equipment Policy":
            return "IT Support"

        # ---------------------------------------------------------
        # SAFETY
        # ---------------------------------------------------------

        if category == "Workplace Safety & Ergonomics":
            return "Administration / Facilities"

        # ---------------------------------------------------------
        # HOSTILE WORKPLACE
        # ---------------------------------------------------------

        if category == "Hostile Work Environment":
            return "Human Resources"

        # ---------------------------------------------------------
        # MANAGERIAL MISCONDUCT
        # ---------------------------------------------------------

        if category == "Managerial Misconduct":
            return "Human Resources"

        # ---------------------------------------------------------
        # POLICY NON-COMPLIANCE
        # ---------------------------------------------------------

        if category == "Policy Non-Compliance":
            return "HR / Compliance"

        # ---------------------------------------------------------
        # ML FALLBACK
        # ---------------------------------------------------------

        return self.department_model.predict([text])[0]

    # =========================================================
    # MAIN PREDICTION
    # =========================================================

    def predict(self, description):

        description = description.strip()

        if not description:
            return {
                "category": "Unknown",
                "severity": "Low",
                "department": "Human Resources"
            }

        # Category
        category = self.classify_category(description)

        # Severity
        severity = self.classify_severity(
            description,
            category
        )

        # Department
        department = self.classify_department(
            description,
            category
        )

        return {
            "category": category,
            "severity": severity,
            "department": department
        }