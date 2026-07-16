# Published by the state machine on /assembly_progress — observability only,
# nothing should sequence off this.
string state              # current state machine state (IDLE, DETECT, ASSIGN_ROLES, ...)
string assembly_name
int32 step                # current step index (0-based), -1 when idle
int32 steps_total
string move_part
string hold_part
string socket_part
string holder_robot
string inserter_robot
int32 attempt             # 1-based attempt number for the current step
string message
