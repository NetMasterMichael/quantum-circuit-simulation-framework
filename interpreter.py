from quantum_circuit import Circuit
from instructions import Opcode, Instruction
import sys

class Interpreter:

    DIRECT_EXECUTE_ARGUMENT = "-e"

    def __init__(self):
        # Read program in
        direct_execution_mode = False
        file_execution_mode = False
        self.arguments = sys.argv[1:]
        self._execution_stack = []
        
        # Start main interpreter loop
        self.main()


    def main(self):
        if (self.arguments[0] == self.DIRECT_EXECUTE_ARGUMENT):
            self.direct_execution_mode = True
            program = self.arguments[1].split("\n")
        else:
            self.file_execution_mode = True
            with open(self.arguments[0], "r") as file:
                contents = file.read()
            program = contents.split("\n")
        
        # Parse the program into a stack
        for line in program:
            opcode = None
            line_tokens = line.split(" ")
            match line_tokens[0]:
                case "\n":
                    continue
                case "":
                    continue
                case "QUBITS":
                    opcode = Opcode.QUBITS
                case "RUNS":
                    opcode = Opcode.RUNS
                case "INIT":
                    opcode = Opcode.INIT
                case "APPLY":
                    opcode = Opcode.APPLY
                case "MEASURE":
                    opcode = Opcode.MEASURE
                case "SHOW":
                    opcode = opcode.SHOW
                case _:
                    print(f"Fatal error: {line_tokens[0]} is not a valid opcode")
                    quit()

            self._execution_stack.append(Instruction(opcode, line_tokens[1]))

        print(self._execution_stack)

if __name__ == "__main__":
    Interpreter()