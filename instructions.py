import enum

class Opcode(enum.Enum):
    QUBITS = "QUBITS"
    RUNS = "RUNS"
    INIT = "INIT"
    APPLY = "APPLY"
    MEASURE = "MEASURE"
    SHOW = "SHOW"

