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
            instruction = None
            line_tokens = line.split(" ")
            match line_tokens[0]:
                case "\n":
                    continue
                case "":
                    continue
                case "QUBITS":
                    instruction = Instruction(Opcode.QUBITS, int(line_tokens[1]))
                case "RUNS":
                    instruction = Instruction(Opcode.RUNS, int(line_tokens[1]))
                case "INIT":
                    instruction = Instruction(Opcode.INIT)
                case "APPLY":
                    # Backend circuit class handles all the parsing, so just pass the whole string as operand
                    instruction = Instruction(Opcode.APPLY, line_tokens[1])
                case "MEASURE":
                    instruction = Instruction(Opcode.MEASURE, line_tokens[1])
                case "SHOW":
                    instruction = Instruction(Opcode.SHOW, line_tokens[1])
                case _:
                    print(f"Fatal error: {line_tokens[0]} is not a valid opcode")
                    quit()

            self._execution_stack.append(instruction)

        print(self._execution_stack)

if __name__ == "__main__":
    Interpreter()