conv1 trace

graph(%self.1 : __torch__.linear.Conv1D,
      %data : Float(4, strides=[1], requires_grad=0, device=cpu),
      %mem : Float(8, strides=[1], requires_grad=0, device=cpu)):
  %linear : __torch__.linear.Linear = prim::GetAttr[name="linear"](%self.1)
  %10 : Tensor[] = prim::ListConstruct(%mem, %data)
  %11 : int = prim::Constant[value=0]() # /Users/chengmo/Work/rnnoise/examples/linear.py:113:0
  %x.1 : Float(12, strides=[1], requires_grad=0, device=cpu) = aten::cat(%10, %11) # /Users/chengmo/Work/rnnoise/examples/linear.py:113:0
  %63 : Tensor = prim::CallMethod[name="forward"](%linear, %x.1)
  %x2 : Float(1, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::mul(%63, %63) # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:49:0
  %27 : Double(requires_grad=0, device=cpu) = prim::Constant[value={0.60863}]() # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:50:0
  %28 : Float(1, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::mul(%x2, %27) # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:50:0
  %29 : Double(requires_grad=0, device=cpu) = prim::Constant[value={96.3924}]() # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:50:0
  %30 : int = prim::Constant[value=1]() # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:50:0
  %31 : Float(1, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::add(%28, %29, %30) # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:50:0
  %32 : Float(1, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::mul(%31, %x2) # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:50:0
  %33 : Double(requires_grad=0, device=cpu) = prim::Constant[value={952.528}]() # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:50:0
  %34 : int = prim::Constant[value=1]() # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:50:0
  %num : Float(1, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::add(%32, %33, %34) # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:50:0
  %36 : Double(requires_grad=0, device=cpu) = prim::Constant[value={11.886}]() # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:51:0
  %37 : Float(1, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::mul(%x2, %36) # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:51:0
  %38 : Double(requires_grad=0, device=cpu) = prim::Constant[value={413.368}]() # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:51:0
  %39 : int = prim::Constant[value=1]() # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:51:0
  %40 : Float(1, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::add(%37, %38, %39) # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:51:0
  %41 : Float(1, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::mul(%40, %x2) # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:51:0
  %42 : Double(requires_grad=0, device=cpu) = prim::Constant[value={952.724}]() # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:51:0
  %43 : int = prim::Constant[value=1]() # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:51:0
  %den : Float(1, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::add(%41, %42, %43) # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:51:0
  %45 : Float(1, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::mul(%63, %num) # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:52:0
  %y : Float(1, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::div(%45, %den) # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:52:0
  %47 : float = prim::Constant[value=-1.]() # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:54:0
  %48 : float = prim::Constant[value=1.]() # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:54:0
  %49 : Float(1, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::clamp(%y, %47, %48) # /Users/chengmo/Work/rnnoise/examples/rnnoise_activation.py:54:0
  %50 : int = prim::Constant[value=0]() # /Users/chengmo/Work/rnnoise/examples/linear.py:114:0
  %51 : int = prim::Constant[value=4]() # /Users/chengmo/Work/rnnoise/examples/linear.py:114:0
  %52 : int = prim::Constant[value=9223372036854775807]() # /Users/chengmo/Work/rnnoise/examples/linear.py:114:0
  %53 : int = prim::Constant[value=1]() # /Users/chengmo/Work/rnnoise/examples/linear.py:114:0
  %54 : Float(8, strides=[1], requires_grad=0, device=cpu) = aten::slice(%x.1, %50, %51, %52, %53) # /Users/chengmo/Work/rnnoise/examples/linear.py:114:0
  %55 : NoneType = prim::Constant()
  %56 : Float(8, strides=[1], requires_grad=0, device=cpu) = aten::clone(%54, %55) # /Users/chengmo/Work/rnnoise/examples/linear.py:114:0
  %57 : (Float(1, 8, strides=[8, 1], requires_grad=0, device=cpu), Float(8, strides=[1], requires_grad=0, device=cpu)) = prim::TupleConstruct(%49, %56)
  return (%57)



linear trace 日志 

graph(%self : __torch__.linear.Linear,
      %x.1 : Float(1, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %bias : Tensor = prim::GetAttr[name="bias"](%self)
  %weights : Tensor = prim::GetAttr[name="weights"](%self)
  %6 : int = prim::Constant[value=0]() # /Users/chengmo/Work/rnnoise/examples/linear.py:53:0
  %7 : int = aten::size(%weights, %6) # /Users/chengmo/Work/rnnoise/examples/linear.py:53:0
  %8 : Long(device=cpu) = prim::NumToTensor(%7)
  %12 : int = aten::Int(%8)
  %13 : int = prim::Constant[value=-1]() # /Users/chengmo/Work/rnnoise/examples/linear.py:53:0
  %14 : int[] = prim::ListConstruct(%13, %12)
  %x : Float(1, 4, strides=[4, 1], requires_grad=0, device=cpu) = aten::reshape(%x.1, %14) # /Users/chengmo/Work/rnnoise/examples/linear.py:53:0
  %16 : Float(1, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::matmul(%x, %weights) # /Users/chengmo/Work/rnnoise/examples/linear.py:54:0
  %17 : int = prim::Constant[value=1]() # /Users/chengmo/Work/rnnoise/examples/linear.py:54:0
  %18 : Float(1, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::add(%16, %bias, %17) # /Users/chengmo/Work/rnnoise/examples/linear.py:54:0
  return (%18)