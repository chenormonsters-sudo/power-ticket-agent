class PowerAgentBaseError(Exception):
    """所有 EduAgent 自定义异常的基类。"""
    def __init__(self, message: str, agent_type: str = "", details: dict = None):
        super().__init__(message)
        self.agent_type = agent_type
        self.details = details or {}


class LLMAPIError(PowerAgentBaseError):
    """大模型 API 调用失败（超时/限流/网络错误）。属于【可重试】异常。"""
    pass


class DatabaseError(PowerAgentBaseError):
    """连接超时，属于可重试"""
    pass

class OperationTicketError(PowerAgentBaseError):
    """。"""
    pass

class InvalidInputError(PowerAgentBaseError):
    """用户输入不合法。属于【不可重试】异常。"""
    pass


class AuthenticationError(PowerAgentBaseError):
    """认证失败。属于【不可重试】异常。"""
    pass



if __name__ == "__main__":
    try:
        raise LLMAPIError("DeepSeek 超时", agent_type="review", details={"timeout": 30})
    except PowerAgentBaseError as e:
        print(f"捕获: {type(e).__name__}")
        print(f"  消息: {e}")
        print(f"  Agent: {e.agent_type}")
        print(f"  详情: {e.details}")
