"""Gemini agent with tool-calling for flight and shopping search."""

from __future__ import annotations

import json
import time
from typing import Any

import google.generativeai as genai

from .config import Config
from .tools.serpapi import search_flights, search_shopping

from datetime import date

TODAY = date.today()

SYSTEM_PROMPT = f"""Bạn là Tara Bot — một agent thông minh chuyên tìm kiếm chuyến bay và săn giá đồ.

NGUYÊN TẮC:
- Trả lời bằng tiếng Việt tự nhiên, thân thiện.
- Khi user hỏi vé máy bay, gọi tool search_flights.
- Khi user hỏi giá sản phẩm, gọi tool search_shopping.
- Sau khi tool trả kết quả, chuyển tiếp NGUYÊN VĂN kết quả đó cho user, chỉ thêm 1-2 câu ngắn ở đầu hoặc cuối.
- KHÔNG reformat lại kết quả từ tool — giữ nguyên định dạng.
- Có thể nói chuyện thông thường (chào hỏi, tạm biệt) — không cần gọi tool.

Hôm nay là {TODAY.strftime("%A, %d/%m/%Y")} — ĐÂY LÀ MỐC THỜI GIAN HIỆN TẠI.
Mặc định cho các câu hỏi mơ hồ về thời gian:
- "cuối tuần" → thứ Sáu tuần gần nhất (không quá khứ)
- "tuần sau" → tuần tiếp theo
- Nếu không rõ, lấy ngày đi và ngày về hợp lý."""

TOOL_FUNCTIONS: dict[str, Any] = {
    "search_flights": search_flights,
    "search_shopping": search_shopping,
}

GEMINI_TOOLS = [
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="search_flights",
                description="Tìm chuyến bay. Trả về giá, hãng, giờ bay.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "departure_id": genai.protos.Schema(type=genai.protos.Type.STRING, description="Mã sân bay đi (IATA). Mặc định SGN"),
                        "arrival_id": genai.protos.Schema(type=genai.protos.Type.STRING, description="Mã sân bay đến (IATA)"),
                        "outbound_date": genai.protos.Schema(type=genai.protos.Type.STRING, description="Ngày đi (YYYY-MM-DD)"),
                        "return_date": genai.protos.Schema(type=genai.protos.Type.STRING, description="Ngày về (YYYY-MM-DD)"),
                        "adults": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Số người lớn. Mặc định 1"),
                    },
                    required=["arrival_id"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name="search_shopping",
                description="Tìm sản phẩm, so sánh giá.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "query": genai.protos.Schema(type=genai.protos.Type.STRING, description="Tên sản phẩm cần tìm"),
                    },
                    required=["query"],
                ),
            ),
        ]
    )
]


class Agent:
    def __init__(self):
        genai.configure(api_key=Config.gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT,
            tools=GEMINI_TOOLS,
        )
        self.chat_session = self.model.start_chat(history=[])

    def chat(self, user_message: str) -> str:
        """Send user message, execute tool calls if needed, return response."""
        for iteration in range(5):
            if iteration == 0:
                response = self.chat_session.send_message(user_message)
            else:
                response = self.chat_session.send_message(
                    genai.protos.Content(parts=tool_response_parts, role="user")
                )

            parts = response.candidates[0].content.parts
            text_parts = []
            tool_calls = []

            for part in parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
                if hasattr(part, "function_call") and part.function_call.name:
                    tool_calls.append(part.function_call)

            if not tool_calls:
                return "".join(text_parts) or "Xin lỗi, em không hiểu yêu cầu. Thử lại nhé!"

            tool_response_parts = []
            for call in tool_calls:
                result = self._execute_tool(call.name, dict(call.args))
                tool_response_parts.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=call.name,
                            response={"result": result},
                        )
                    )
                )

        return "Xin lỗi, em không thể xử lý yêu cầu này ngay bây giờ. Thử lại với câu hỏi đơn giản hơn nhé!"

    def _execute_tool(self, name: str, args: dict) -> str:
        fn = TOOL_FUNCTIONS.get(name)
        if not fn:
            return f"Unknown tool: {name}"
        try:
            return fn(**args)
        except Exception as e:
            return f"Lỗi khi chạy {name}: {e}"
