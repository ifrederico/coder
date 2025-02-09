# src/memory_store.py

class MemoryStore:
    """
    Keeps a list of { 'req': str, 'code': str, 'passed': bool } records.
    """
    def __init__(self):
        self.records = []

    def add_record(self, req, code, passed):
        self.records.append({"req": req, "code": code, "passed": passed})

    def get_positive_samples(self):
        return [(r["req"], r["code"]) for r in self.records if r["passed"]]

    def get_negative_samples(self):
        return [(r["req"], r["code"]) for r in self.records if not r["passed"]]

    def all_records(self):
        return self.records
